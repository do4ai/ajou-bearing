"""세 방법론 결과 비교 시각화"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent / 'results'
methods = {
    '경민팀\n(Order Tracking\n+CNN-BiLSTM)': BASE / '01_경민팀' / 'lobo_results.csv',
    '태환팀\n(Spectrogram\n+2D CNN-LSTM)':   BASE / '02_태환팀' / 'lobo_results.csv',
    '교수 설계\n(Fast Kurtogram\n+Ensemble)': BASE / '03_교수설계' / 'lobo_results.csv',
}

rows = []
for label, path in methods.items():
    if path.exists():
        df = pd.read_csv(path)
        rows.append({'method': label,
                     'mean_rmse':  df['rmse_s'].mean(),
                     'mean_score': df['asym_score'].mean(),
                     **{f"score_{r['val_bearing']}": r['asym_score']
                        for _, r in df.iterrows()}})

cdf = pd.DataFrame(rows)
print("\n" + "="*70)
print("  방법론 비교 최종 결과")
print("="*70)
print(cdf[['method','mean_rmse','mean_score']].to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#4472C4', '#ED7D31', '#70AD47']
labels = [m.replace('\n', ' ') for m in cdf['method']]

axes[0].bar(labels, cdf['mean_rmse'], color=colors, edgecolor='white', linewidth=1.5)
axes[0].set_ylabel('Mean RMSE (s)', fontsize=12)
axes[0].set_title('RMSE 비교 (낮을수록 좋음)', fontsize=12)
axes[0].tick_params(axis='x', labelsize=8)

axes[1].bar(labels, cdf['mean_score'], color=colors, edgecolor='white', linewidth=1.5)
axes[1].set_ylim(0, 1.05)
axes[1].set_ylabel('Mean Asymmetric Score', fontsize=12)
axes[1].set_title('비대칭 Score 비교 (높을수록 좋음)', fontsize=12)
axes[1].tick_params(axis='x', labelsize=8)

plt.tight_layout()
plt.savefig(BASE / 'comparison.png', dpi=150)
print(f"\n  비교 그래프 저장: {BASE / 'comparison.png'}")
