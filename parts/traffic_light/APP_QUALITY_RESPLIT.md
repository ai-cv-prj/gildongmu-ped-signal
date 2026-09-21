# 앱 화질 데이터 재분할

기존 train·val·test 총 36,808장을 합쳐 약 80:10:10으로 새로 분할했다. 신호등/횡단보도 라벨 개수, 신호등 크기 구간, 이전 분할의 구성 비율을 함께 고려했다. 원본 분할·라벨·모델은 보존했다. 사용자가 실행한 50 epoch 학습이 완료됐으며, 2026-09-21에 48 epoch의 best.pt로 새 val/test 전체를 평가했다. 결과는 [재분할 모델 평가](RESPLIT_EVALUATION_20260921.md)에 있다.

| 분할 | 이미지 | 신호등 박스 | 횡단보도 박스 | 신호등 2개 이상 라벨링된 이미지 |
|---|---:|---:|---:|---:|
| train | 29,447 | 11,623 | 29,689 | 251 |
| val | 3,681 | 1,450 | 3,710 | 29 |
| test | 3,680 | 1,449 | 3,710 | 30 |

기존 train의 복수 신호등 이미지는 9장이었다. 새 train에는 이전 val 이미지 446장과 이전 test 이미지 35장이 포함됐다. 모든 이미지는 정확히 한 분할에만 배정했다.

## 입력과 라벨 보존

- 이미 준비된 Chromium canvas 긴 변 최대 960px/JPEG quality=0.8 이미지를 그대로 링크한다. 재압축하거나 원본 화질로 되돌리지 않았다.
- 정규화 YOLO 라벨은 바이트 단위로 동일하게 복사했다. 이미지 크기가 바뀌지 않았으므로 좌표 변환은 필요 없다.
- 전체 이미지 축소·압축 후 만들었던 분류 crop 14,522개도 그대로 링크하고, 원본 전체 이미지와 같은 split에 배치했다. 색상 분류기 학습은 실행하지 않았다.
- 링크가 참조하는 `datasets/app_quality_960_q08_v1/` 폴더는 유지해야 한다. 이를 삭제하면 새 데이터셋 이미지/crop 링크가 끊어진다.

## 분할 방식과 한계

seed=42로 재현 가능하게 배정했다. 동일 원본 SHA-256과 파일명 접두사/번호 100개 구간으로 묶은 그룹은 분할 사이에 나누지 않았다. 총 10,973개 그룹, 최대 그룹 크기는 79장이다. 그룹 보존 때문에 이미지 수는 정확한 80:10:10과 소폭 다를 수 있다.

촬영 장소·세션 메타데이터는 확인되지 않았다. 파일번호 구간은 인접 이미지가 양쪽에 들어갈 가능성을 줄이는 휴리스틱이며, 같은 장소·유사 장면이 완전히 분리됐다는 보장은 아니다. 확인된 촬영 그룹이 생기면 그 기준으로 재분할하는 것이 더 정확하다.

새 test에는 기존 train 이미지 3,619장이 포함됐다. 따라서 기존 체크포인트의 새 test 점수는 독립 평가 기준으로 사용할 수 없다. 원본 사전학습 가중치에서 새 train으로 학습한 모델의 새 test 결과를 평가해야 한다. 이전 test 45장 결과와 새 test 3,680장 결과도 평가 대상이 달라 점수를 직접 비교해 개선 폭으로 해석하지 않는다.

## 학습 실행

현재 경로의 학습은 완료됐으므로 아래 명령을 같은 출력 경로에 다시 실행하지 않는다. 다른 실행을 재현할 때는 새 출력 경로를 사용한다.

프로젝트 루트에서 기존 `.venv`를 사용한다. 추가 패키지는 필요 없다. 환경은 기존 실험과 같은 ultralytics 8.4.150, torch 2.11.0+cu128, torchvision 0.26.0+cu128, opencv-python 4.14.0.94, Pillow 12.3.0, PyYAML 6.0.3을 확인하도록 되어 있다.

```bash
cd /home/user/ai_cv_prj
.venv/bin/python -m parts.traffic_light.tools.resplit_app_quality train
```

사용자가 앞서 요청한 최대 50 epoch를 적용했다. patience=7은 유지했으므로 검증 성능이 개선되지 않으면 50회 이전에 종료될 수 있다. 시작 가중치는 기존 파인튜닝 모델이 아닌 `yolo26s.pt`이며 `resume=False`다. batch=16, imgsz=960, seed=42 및 나머지 설정은 기존 검출기 설정 사본을 재사용한다. 분할과 최대 epoch가 달라졌으므로 이전 실험 대비 순수한 분할 효과만을 통제한 실험은 아니다.

학습 없이 설정만 확인하려면:

```bash
.venv/bin/python -m parts.traffic_light.tools.resplit_app_quality train --print-only
```

학습은 이미지/라벨 및 설정의 해시를 먼저 확인한다. 기존 학습 출력이 존재하면 덮어쓰지 않고 중단한다. 이미 완료한 재분할을 다시 실행할 필요는 없다.

## 학습 완료 후 test 평가 명령

실제 완료한 평가는 다음 명령으로 재현한다. 공식 AP와 confidence=0.4 precision/recall, 작은 신호등 recall 및 이미지별 예측을 함께 저장한다. 출력에는 새 경로를 사용한다.

```bash
.venv/bin/python -m parts.traffic_light.tools.evaluate_dataset_splits \
  --resplit-config parts/traffic_light/configs/app_quality_resplit.json \
  --splits val test --models app \
  --output runs/app_quality_960_q08_resplit_v1/evaluation_new
```

결과의 `official` P/R은 validator가 F1 기준으로 선택한 confidence에서의 값이다. `operational`에는 고정 confidence=0.4 지표를 별도로 저장한다. 임계값을 고를 때는 val을 사용하고 test 점수로 설정을 맞추지 않는다. `--resplit-config`를 생략하면 기본 설정이 이전 분할을 가리키므로 반드시 지정한다. 재분할 평가는 기존 학습 이미지가 포함된 test에서의 부적절한 비교를 막기 위해 `--models app`만 허용한다.

## 파일 위치

- 재분할 설정: `parts/traffic_light/configs/app_quality_resplit.json`
- 재분할·학습 코드: `parts/traffic_light/tools/resplit_app_quality.py`
- 새 데이터 YAML: `datasets/app_quality_960_q08_resplit_v1/detector_app/data.yaml`
- 분할 집계·이전→새 분할 표: `datasets/app_quality_960_q08_resplit_v1/summary.json`
- 원본 이미지별 이전/새 split 및 그룹: 같은 데이터셋의 `detector_app/manifest.jsonl`
- 무결성 점검: 같은 데이터셋의 `integrity_audit.json`
- 학습 설정 미리보기: 같은 데이터셋의 `train_args_preview.json`
- 학습 시 결과: `runs/app_quality_960_q08_resplit_v1/detector/app/`
- 학습 시 최적 가중치: 위 폴더의 `weights/best.pt`
- 학습 시 설정/가중치 해시 기록: `runs/app_quality_960_q08_resplit_v1/audit/detector_app/`

완료한 검증: 단위·통합 테스트 5개 통과, 이미지 36,808장 전부 보존, 라벨 전부 바이트 동일, 원본 이미지 해시/그룹의 분할 간 중복 0, crop 14,522개 전부 원본 전체 이미지와 split 일치. 이 검증은 데이터 재분할 검증이며 새 모델의 성능을 뜻하지 않는다.
