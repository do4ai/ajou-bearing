"""태환 팀: VMD + STFT Spectrogram + 2D CNN-LSTM Attention (RUL in hours)"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import sys, warnings, time
warnings.filterwarnings('ignore')
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from shared_utils import asym_score

import numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import stft, butter, filtfilt
from scipy.stats import kurtosis as sp_kurt
from scipy.ndimage import zoom
from sklearn.metrics import mean_squared_error
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR   = Path(ROOT) / 'data' / 'raw'
RESULT_DIR = Path(ROOT) / 'results' / '02_태환팀'
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FS=25600; TRAIN_NAMES=['Train1','Train2','Train3','Train4']
EPOCHS,LR,SEQ_LEN=60,2e-4,8; SPEC_H,SPEC_W=32,32

print("="*65); print("  태환 팀: VMD + STFT Spectrogram + 2D CNN-LSTM"); print("="*65)

# ── 신호처리 ─────────────────────────────────────────────────────
def vmd_approx(sig, K=4, fs=FS):
    nyq=fs/2; bounds=np.linspace(0,nyq,K+1); modes=[]
    for k in range(K):
        lo=max(bounds[k],1); hi=min(bounds[k+1],nyq*0.99)
        b,a=butter(4,[lo/nyq,hi/nyq],btype='band')
        modes.append(filtfilt(b,a,sig))
    return np.array(modes)

def spectrogram(sig, fs=FS, nfft=256, hop=128, fmax=5000, H=32, W=32):
    f,_,Z=stft(sig,fs=fs,nperseg=nfft,noverlap=nfft-hop)
    mask=f<=fmax; S=np.abs(Z[mask])
    S=20*np.log10(S+1e-10); S=(S-S.min())/(S.max()-S.min()+1e-10)
    return zoom(S.astype(np.float32),(H/S.shape[0],W/S.shape[1]))

def load_specs(name):
    d=DATA_DIR/name; sigs=np.load(d/'vibration.npy'); op=pd.read_csv(d/'operating.csv')
    base=[]; specs=[]
    for i in range(min(5,len(sigs))):
        modes=vmd_approx(sigs[i,0].astype(np.float64))
        k_sel=int(np.argmax([sp_kurt(m) for m in modes]))
        base.append(spectrogram(modes[k_sel]))
    baseline=np.mean(base,axis=0)
    print(f"    {name}: {len(sigs)} 측정...")
    for i in range(len(sigs)):
        modes=vmd_approx(sigs[i,0].astype(np.float64))
        k_sel=int(np.argmax([sp_kurt(m) for m in modes]))
        sp=spectrogram(modes[k_sel])
        sp=np.maximum(sp-baseline,0); sp=(sp-sp.min())/(sp.max()-sp.min()+1e-10)
        specs.append(sp[np.newaxis])   # [1,H,W]
    return np.array(specs,dtype=np.float32), op

# ── 모델 ─────────────────────────────────────────────────────────
class DS(Dataset):
    def __init__(self,X,y): self.X=torch.tensor(X,dtype=torch.float32); self.y=torch.tensor(y,dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i],self.y[i]

class CNN2D(nn.Module):
    def __init__(self,H,W):
        super().__init__()
        self.enc=nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),nn.BatchNorm2d(16),nn.GELU(),
            nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.GELU(),nn.MaxPool2d(2),nn.Dropout2d(0.2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.GELU(),nn.AdaptiveAvgPool2d((4,4)))
        self.proj=nn.Sequential(nn.Linear(64*4*4,128),nn.GELU(),nn.Dropout(0.3))
        self.lstm=nn.LSTM(128,64,2,batch_first=True,dropout=0.3)
        self.attn=nn.Sequential(nn.Linear(64,16),nn.Tanh(),nn.Linear(16,1),nn.Softmax(dim=1))
        self.fc=nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Dropout(0.2),nn.Linear(32,1))
    def forward(self,x):
        B,T,C,H,W=x.shape
        e=self.enc(x.view(B*T,C,H,W)).view(B*T,-1)
        p=self.proj(e).view(B,T,128)
        l,_=self.lstm(p); w=self.attn(l); ctx=(l*w).sum(1)
        return self.fc(ctx).squeeze(-1)

def aloss(p,t): e=p-t; return torch.where(e<=0,2.5*e.pow(2),e.pow(2)).mean()

# ── 로딩 ─────────────────────────────────────────────────────────
print("\n[1] VMD + STFT 스펙트로그램 생성...")
all_specs,all_ops={},{}
for nm in TRAIN_NAMES:
    sp,op=load_specs(nm)
    all_specs[nm]=sp; all_ops[nm]=op; all_ops[nm]['rul_h']=op['rul_seconds']/3600

# ── LOBO ─────────────────────────────────────────────────────────
print("\n[2] LOBO + Spectrogram CNN-LSTM...")
results=[]
for val in TRAIN_NAMES:
    tns=[b for b in TRAIN_NAMES if b!=val]; print(f"\n  Fold: Val={val}")
    def mk_seqs(nm,sl=SEQ_LEN):
        sp=all_specs[nm]; rul=all_ops[nm]['rul_h'].values
        s,t=[],[]
        for i in range(len(sp)-sl): s.append(sp[i:i+sl]); t.append(rul[i+sl-1])
        return np.array(s,dtype=np.float32),np.array(t,dtype=np.float32)
    tr_s=np.concatenate([mk_seqs(n)[0] for n in tns])
    tr_t=np.concatenate([mk_seqs(n)[1] for n in tns])
    vl_s,vl_t=mk_seqs(val)
    
    m=CNN2D(SPEC_H,SPEC_W); opt=optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-4)
    sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    tld=DataLoader(DS(tr_s,tr_t),batch_size=8,shuffle=True)
    vld=DataLoader(DS(vl_s,vl_t),batch_size=16)
    best_sc,best_st=-np.inf,None
    for ep in range(1,EPOCHS+1):
        m.train()
        for xb,yb in tld:
            opt.zero_grad(); aloss(m(xb),yb).backward()
            nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            vp=np.concatenate([np.nan_to_num(m(xb).numpy()) for xb,_ in vld])
        s=asym_score(vp,vl_t)
        if s>best_sc: best_sc=s; best_st={k:v.clone() for k,v in m.state_dict().items()}
        if ep%20==0:
            rmse_m=np.sqrt(mean_squared_error(vl_t,vp))*60
            print(f"    Ep{ep:3d}/{EPOCHS} Score={s:.4f} RMSE={rmse_m:.1f}min")
    
    m.load_state_dict(best_st); m.eval()
    with torch.no_grad():
        preds=np.concatenate([np.nan_to_num(m(xb).numpy()) for xb,_ in vld])
    rmse_s=np.sqrt(mean_squared_error(vl_t,preds))*3600; score=asym_score(preds,vl_t)
    results.append({'val_bearing':val,'rmse_s':rmse_s,'asym_score':score})
    print(f"  → RMSE={rmse_s:.0f}s  AsymScore={score:.4f}")
    
    t_h=all_ops[val]['t_seconds'].values[SEQ_LEN:]/3600
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    ax[0].plot(t_h,vl_t,'k-',lw=2,label='True'); ax[0].plot(t_h,preds,'r--',label='Pred')
    ax[0].set(xlabel='Time(h)',ylabel='RUL(h)',title=f'{val} Spectrogram CNN-LSTM'); ax[0].legend()
    sp_last=all_specs[val][-1,0]
    ax[1].imshow(sp_last,aspect='auto',origin='lower',cmap='viridis')
    ax[1].set(title=f'{val} Spectrogram (last)',xlabel='Time frames',ylabel='Freq bins')
    plt.tight_layout(); plt.savefig(RESULT_DIR/f'{val}.png',dpi=120); plt.close()

res=pd.DataFrame(results)
print("\n"+"="*65); print("  태환 팀 최종 결과"); print("="*65)
print(res.to_string(index=False))
print(f"\n  평균 RMSE:      {res.rmse_s.mean():.0f} s")
print(f"  평균 AsymScore: {res.asym_score.mean():.4f}")
res.to_csv(RESULT_DIR/'lobo_results.csv',index=False); print(f"\n  저장: {RESULT_DIR}")
