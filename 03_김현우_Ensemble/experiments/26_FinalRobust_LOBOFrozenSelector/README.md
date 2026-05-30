# 26_FinalRobust_LOBOFrozenSelector

Final test용 실험 scaffold.

목표:

- 현재 public Validation/Test1~6에 과적합된 수동 판단을 배제한다.
- Train1~4 LOBO에서 고정된 규칙만 사용한다.
- 최종 테스트셋이 공개되면 같은 inference 절차로 제출 파일을 생성한다.

핵심 정책:

- anchor는 `5_HIBlend_Baseline_ChannelSym`.
- EOL gate는 fixed threshold만 사용한다.
- trajectory KNN은 fixed `k=12`, fixed window `[10, 20]`, fixed feature list만 사용한다.
- final Test 개별 bearing 이름별 수동 보정 금지.

현재 구현 상태:

- 최종 테스트 데이터 포맷이 아직 공개되지 않았으므로 scaffold만 생성.
- 데이터 공개 후 `experiment.py`에서 final test bearing list를 받아 inference하도록 구현 예정.
