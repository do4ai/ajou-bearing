# CLAUDE.md — 프로젝트 가이드

## 프로젝트
KSPHM-KIMM 2026 기계 데이터 챌린지 참가 프로젝트. NSK 30306 테이퍼 롤러 베어링의 잔여수명(RUL) 예측.

## 데이터
- **원본**: KIMM 데이터 플랫폼에서 다운로드 (TDMS 포맷, Train.zip 7.3GB + Test.zip 4.77GB)
- **변환**: `data/convert_tdms.py` 로 TDMS → `data/raw/{Train1..Val2}/vibration.npy + operating.csv`
- **합성 데이터**: `data/synthetic_generator.py` (실제 데이터 없을 때 테스트용)
- 샘플링: 진동 25.6kHz 4채널, 운전조건 0.1Hz, 10분 주기 1분 취득

## 구조
- `shared/utils.py`: 공통 함수 (asym_score, load_bearing, TRAIN_NAMES, FS, ORDERS)
- 3개 방법론 폴더, 각각 `pipeline.py` 하나로 전체 실행
- 각 폴더에 `data/`, `models/`, `results/` 포함
- `.gitignore`: *.pt, *.npy, *.npz, *.tdms 제외

## 평가
- **asym_score**: 퍼센트 오차 기반 비대칭 점수 (늦은 예측 2.5x 패널티)
- Er = 100 × (ActRUL - PredRUL) / ActRUL
- RUL 단위: **초(seconds)**

## 실행 시 주의사항
- PyTorch를 XGBoost보다 먼저 import 해야 함 (OpenMP 충돌 방지)
- `KMP_DUPLICATE_LIB_OK=TRUE`, `OMP_NUM_THREADS=1` 필수 (shared/utils.py에 설정됨)
- RUL 타겟을 max로 나누어 정규화 후 학습, 평가 시 다시 곱해서 복원
- GPU 사용 가능하면 자동 감지됨 (현재 CPU 기준으로 작성됨)

## 챌린지 일정
- 예비 제출: 6/1~6/5
- 최종 제출: 6/8
- 제출물: 팀이름_validation.xlsx, 팀이름_code.zip, 팀이름_report.pdf
