# gildongmu-ped-signal

보행자 신호등·횡단보도 검출과 신호등 색상 분류 실험 저장소입니다.

- [현재 진행 상황](parts/traffic_light/PROGRESS.md)
- [재분할 데이터로 학습 시작하기](parts/traffic_light/APP_QUALITY_RESPLIT.md)
- [최신 재분할 모델 평가](parts/traffic_light/RESPLIT_EVALUATION_20260921.md)
- [기존 분할 전체 평가](parts/traffic_light/DATASET_SPLIT_EVALUATION_20260919.md)
- [평가 결과 상세 리뷰](parts/traffic_light/DATASET_SPLIT_REVIEW_20260919.md)
- [추론·데이터 준비 도구 안내](parts/traffic_light/README.md)

2026-09-21 기준 재분할 모델의 50 epoch 학습과 새 validation/test 전체 평가까지 완료했습니다.
최적 체크포인트는 48 epoch이며, mAP50–95는 validation 84.41%, test 84.10%입니다.
이미지·가중치·대용량 실행 결과는 Git에서 제외하며, 주요 지표와 재분할 요약은
[reports/20260921](parts/traffic_light/reports/20260921/README.md)에 기록합니다.
