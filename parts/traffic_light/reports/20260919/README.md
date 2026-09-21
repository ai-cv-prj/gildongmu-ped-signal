# 2026-09-19 실험·정리 기록

대용량 `datasets/`, `runs/`와 가중치는 Git에서 제외한다. 다른 환경에서도 수치와 진행 상황을
확인할 수 있도록 최종 JSON의 작은 사본을 이 폴더에 보존한다. 이미지나 프레임별 예측 전체는 포함하지 않는다.

| 파일 | 의미 |
|---|---|
| [dataset_split_metrics.json](dataset_split_metrics.json) | 기존 분할 train/val/test 전체의 기존/앱 화질 검출기 지표·실행 설정·가중치 해시 |
| [dataset_split_diagnostics.json](dataset_split_diagnostics.json) | 크기별 recall, 같은 GT의 검출 변화, 이미지 단위 누락 집계 |
| [dataset_split_integrity.json](dataset_split_integrity.json) | 저장된 예측 재집계, 분할 간 원본 이미지 중복, GT 좌표 검사 |
| [public_test_pipeline_metrics.json](public_test_pipeline_metrics.json) | 공개 test 45장의 batch=1 검출+기존 색상 분류기 평가 |
| [resplit_summary.json](resplit_summary.json) | 새 분할 구성·그룹·manifest 해시·설정 |
| [resplit_integrity.json](resplit_integrity.json) | 재분할 직후 이미지·라벨·crop 보존 검사; training_started=false는 검사 당시 상태 |
| [training_status.json](training_status.json) | 이번 문서 정리 중 확인한 새 분할 학습 진행 상태 |
| [cleanup.json](cleanup.json) | 삭제한 로컬 산출물, 크기, 보존 대상, 각 JSON의 원본 경로 |

두 평가의 수치는 섞어서 비교하지 않는다. `dataset_split_metrics.json`은 batch=16 공식
Ultralytics AP와 confidence=0.4 검출 지표이며, `public_test_pipeline_metrics.json`은 batch=1
추론·별도 AP 집계·색상 분류를 포함한다. 모두 기존 공개 분할 평가이며 실제 앱 촬영 정답 평가는 아니다.

## 문서 정리 시 확인한 상태

- 앱 화질 전처리와 기존 분할 검출기 학습·평가 완료.
- 새 분할 생성 완료: train 29,447장, val 3,681장, test 3,680장.
- 재분할을 전달한 뒤 사용자가 학습을 시작함. 최대 50 epoch·patience=7, 문서 정리 당시 실행 중.
- 앱 화질 색상 분류기 재학습 및 정답 라벨 기반 실제 앱 평가는 미실행.
- 전체 테스트 64개 통과. 실제 Chromium으로 소량 합성 이미지를 인코딩하는 테스트 포함.
- 진행 중인 학습 프로세스를 재실행·중단하거나 데이터/가중치를 변경하지 않음.

## 정리한 파일

- 미완료 전처리 폴더 `datasets/app_quality_960_q08_v1.interrupted_20260918_163608_dfd4ff64/`.
- 정확도 결과가 최종 실행과 같은 최초 공개 test 평가 폴더 `runs/app_quality_960_q08_v1/evaluation_public_test_20260919/`.
- JSON 저장 오류로 중단된 평가 폴더 `runs/app_quality_960_q08_v1/evaluation_splits_20260919/`.
- 기존 작업에서 삭제된 `FINETUNING_REPORT.html`의 삭제를 반영한다. 이전 학습 이력은
  `PROGRESS.md`에 남기고 최신 평가·리뷰 문서를 연결했다.

세 로컬 산출물 폴더에서 합계 1,283,217,541바이트(약 1.28GB)를 정리했다. 삭제 전
141,852개 심볼릭 링크를 확인했고 삭제 대상 참조는 없었다. 완료된 앱 화질 데이터, 새 분할,
원본 라벨·가중치, 최종 평가 결과와 실행 중인 학습 출력은 보존한다.

현재 도구는 전처리, 분할, 학습, 공개 평가, 실제 앱 holdout 평가로 역할이 다르므로 유지했다.
루트 `AGENTS.md`, `.agents/`와 별도 프로젝트 개요 HTML은 로컬 파일로 보존하며 이번 커밋에서 제외한다.
