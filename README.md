# gildongmu-ped-signal

보행자 신호등·횡단보도 검출과 신호등 색상 분류 실험 저장소입니다.

- [현재 진행 상황](parts/traffic_light/PROGRESS.md)
- [재분할 데이터로 학습 시작하기](parts/traffic_light/APP_QUALITY_RESPLIT.md)
- [기존 분할 전체 평가](parts/traffic_light/DATASET_SPLIT_EVALUATION_20260919.md)
- [평가 결과 상세 리뷰](parts/traffic_light/DATASET_SPLIT_REVIEW_20260919.md)
- [추론·데이터 준비 도구 안내](parts/traffic_light/README.md)

2026-09-19 기준 앱 화질 전처리, 기존 분할 검출기 재학습과 비교 평가,
전체 데이터의 새 train/val/test 분할까지 완료했습니다. 사용자가 새 분할의 학습을 시작했으며,
문서 정리 시점에는 실행 중이고 완료 결과는 아직 없습니다.
이미지·가중치·대용량 실행 결과는 Git에서 제외하며, 주요 지표와 재분할 요약은
[reports/20260919](parts/traffic_light/reports/20260919/README.md)에 기록합니다.
