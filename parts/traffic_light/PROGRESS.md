# 신호등 파트 진행 문서

## 현재 상태 — 앱 화질 재학습·평가·재분할 (2026-09-19)

| 단계 | 상태 | 근거/산출물 |
|---|---|---|
| 실제 canvas JPEG 전처리 | 완료 | 36,808장, 색상 crop 14,522개 |
| 기존 분할에서 검출기 재학습 | 완료 | 원본 `yolo26s.pt` 시작, 최대 30 epoch 중 26에서 조기 종료, best 19 epoch |
| 기존/새 검출기 train·val·test 비교 | 완료 | 36,204 / 559 / 45장 전체 실제 추론 |
| 복수 신호등·크기 분포를 고려한 재분할 | 완료 | 29,447 / 3,681 / 3,680장, 원본 보존 |
| 새 분할에서 검출기 학습 | **실행 중** | 사용자가 시작, 최대 50 epoch·patience=7, 완료 기록 없음 |
| 앱 화질 색상 분류기 재학습 | 미실행 | 이전 분류기 유지, 새 분할 crop 배정만 완료 |
| 정답 라벨 기반 실제 앱 촬영 평가 | 미실행 | 앱 평가용 정답 manifest 미확보 |

기존 분할의 전체 mAP50–95는 기존 → 앱 화질 모델 기준 train 85.07→86.20%,
val 65.56→69.91%, test 52.51→55.53%다. test에서 confidence=0.4 기준 신호등 recall은
30.00→39.17%, 작은 신호등 recall은 17.71→28.12%, 횡단보도 recall은 38.02→45.45%다.
신호등 test AP50–95는 54.59→54.08%로 소폭 하락했다. 전체 점수 개선이 모든 지표의
개선을 뜻하지 않으며, 학습 데이터와 test 성능의 차이가 여전히 크다.

신호등 라벨이 2개 이상인 이미지는 기존 train 9장, val 257장, test 44장이었다.
전체 데이터를 합친 뒤 원본 이미지 해시와 파일번호 구간 그룹을 유지하고 라벨 개수·크기를
고려해 배분했다. 새 train에는 복수 신호등 이미지 251장, val 29장, test 30장이 들어갔다.
파일번호 그룹은 촬영 장소·세션을 검증한 메타데이터가 아니므로 유사 장면 누수 가능성은 남는다.
새 test에는 이전 train 이미지가 포함되므로 기존 체크포인트와의 독립 성능 비교에는 사용할 수 없다.

다음 명령으로 사용자가 학습을 시작했다. 문서 정리 시 실행 프로세스와 `args.yaml`, audit를
확인했으며 완료 기록은 없다. **진행 중에는 중복 실행하지 않는다.** 기존 파인튜닝 모델을
이어 학습하지 않는다.

```bash
.venv/bin/python -m parts.traffic_light.tools.resplit_app_quality train
```

완료된 앱 화질 데이터 `datasets/app_quality_960_q08_v1/`와 새 분할
`datasets/app_quality_960_q08_resplit_v1/`를 모두 유지한다. 새 데이터셋은 기존 압축 이미지와
crop에 링크하므로 첫 폴더를 삭제하면 사용할 수 없다. 원본 라벨은 좌표 변경 없이 복사했다.
최적 모델의 예정 경로는 `runs/app_quality_960_q08_resplit_v1/detector/app/weights/best.pt`다.

- 현재 실행 안내: [재분할 학습](APP_QUALITY_RESPLIT.md)
- 기존 실험 재현: [전처리·학습·앱 holdout 평가](APP_QUALITY_EXPERIMENT.md)
- 실제 측정 결과: [전체 분할 평가](DATASET_SPLIT_EVALUATION_20260919.md), [상세 리뷰](DATASET_SPLIT_REVIEW_20260919.md)
- 이전 batch=1 검출+색상 평가: [공개 test 평가](APP_QUALITY_EVALUATION_20260919.md)
- Git에 보존하는 수치·정리 기록: [실험 기록](reports/20260919/README.md)

아래 내용은 이전 실험 이력이다. 현재 작업 상태는 위 표를 기준으로 확인한다.

## 휴대폰 실측 결과와 재학습 준비 (2026-09-17)

기존 공개 데이터의 신호등·횡단보도 2클래스 YOLO와 MobileNet 색상 분류기를 별도
`gildongmu-test-app`에 연결해 휴대폰 촬영으로 확인했다. 실제 YOLO 모델을 사용한
6개 세션의 655프레임에서 보행자 신호등 검출 수는 0개 180프레임, 1개 475프레임,
2개 이상 0프레임이었다. 따라서 다중 신호등에서 횡단보도 소실점·크기로 후보를
고르는 경로는 이 촬영에서 검증되지 않았다. 이 수치는 정답 라벨 없이 집계한 모델
출력 횟수이며 검출 정확도나 재현율이 아니다.

`20260917_160815_galaxy-quantum-3_traffic` 세션의 83번 프레임에는 빨강·초록
보행 신호가 보이지만, 앱의 YOLO confidence 0.40 기준에서는 신호등이 0개였다.
동일 이미지를 낮은 기준으로 재실행하면 초록 신호만 confidence 약 0.39로 나오고
빨간 신호는 0.01 기준에서도 나오지 않았다. 같은 세션 161프레임에서 기존
신호등 전용 모델은 0개 검출 10프레임, 2클래스 모델은 43프레임이었다.
이는 추가 학습 후 검출 약화 가능성을 보여주지만, 두 모델의 정확도 비교에는
동일 프레임의 신호등 전체에 정답 박스를 붙인 평가가 필요하다.

후속 작업은 직접 촬영한 실패·성공 장면에 `pedestrian_signal` 및 `crosswalk`
정답 박스를 모두 붙이고, 원래 데이터와 합쳐 2클래스 모델을 재학습하는 것이다.
별도로 사전학습 `yolo26s.pt`에서 두 클래스를 동시에 파인튜닝하는 비교 실행의
산출물이 `runs/traffic_light_joint_from_yolo26s/`에 생성되었다. 이 실행의
완료 여부와 성능은 아직 확정하지 않았으며, 기존 모델과 같은 라벨이 붙은
휴대폰 평가셋으로 클래스별 검출 재현율과 오검출을 비교해야 한다.

로컬 테스트 영상 `data/`와 내용이 없던 중첩 Git 폴더 `gildongmu-ped-signal/`은
삭제했다. 영상은 Git 관리 대상이 아니므로 과거 영상 테스트를 재현하려면 별도의
원본 영상이 필요하다. 현재 결과만으로 보행자에게 횡단 시작을 안내해서는 안 된다.

## 횡단보도 연결 실험 (2026-09-17)

2클래스 YOLO의 `crosswalk` 검출을 `runtime/pipeline.py`와 `tools/benchmark_yolo_classifier.py`에서 사용할 수 있도록
`--associate-crosswalk` 옵션을 추가했다. 신호등이 1개면 그 신호등만 분류해
`single_signal`로 기록하고, 횡단보도와의 관계는 미확인으로 남긴다. 2개 이상이면 화면
아래·중앙의 횡단보도, 선분 교차로 추정한 소실점, 신호등의 방향·크기, 영상 프레임 간
같은 박스 유지 여부로 후보를 고른다. 선택된 신호등 하나만 색상 분류·표시한다.
`frames.jsonl`에 `association`과 `crosswalks`를 기록한다. 소실점 추정 실패·후보 모호성·
시간적 불일치는 `unknown`이다. 실제 테스트 사진 1장에서는 횡단보도 2개, 신호등 0개가
검출되어 신호등을 연결할 수 없었다. 연결 정확도는 아직 평가되지 않았다.

실행 예:

```bash
.venv/bin/python parts/traffic_light/tools/benchmark_yolo_classifier.py \
  --source "/mnt/c/Users/10/Desktop/2차플젝/파인튜닝 이후 테스트/완료/갤럭시quantum3_신촌_가로_신호등_C_10.mp4" \
  --detector runs/traffic_light_crosswalk_detector/20260916_173303/weights/best.pt \
  --signal-classes pedestrian_signal --crosswalk-class crosswalk \
  --classifier-model mobilenet_v3_small \
  --classifier-weights runs/traffic_light_classifier/20260915T085300Z_3f63cc9d/mobilenet_v3_small/best.pt \
  --associate-crosswalk --device 0 --warmup 0
```

WSL에서는 Windows `C:\Users\...` 경로 대신 `/mnt/c/Users/...`를 사용하고, 공백이 있는
경로를 큰따옴표로 감싼다. 여러 줄 명령의 마지막 줄에는 줄 연결 문자 `\\`를 붙이지 않는다.

2026-09-17 기준 연결 기능 전용 및 전체 테스트 40개가 통과했다. 이 기능은 횡단보도와
신호등의 공간적 후보를 만들 뿐 횡단 시작 허가를 결정하지 않는다. 실사용 전에는 다중
신호등 영상에 횡단보도↔신호등 정답을 붙여 연결 정확도와 `unknown` 비율을 측정해야 한다.

## 폴더 역할 분리

```text
parts/traffic_light/
├── runtime/   # 서버가 import하는 실시간 추론·연결 로직
├── tools/     # 데이터 준비·파인튜닝·영상 벤치마크 CLI
└── tests/     # 두 영역의 자동 테스트
```

모델을 다시 학습할 때는 `tools/`를 실행하고, 프로젝트 서버에서는
`parts.traffic_light.runtime.pipeline`의 함수만 호출한다. 기존 `data/`의 통합 테스트용
영상 5개(약 243MB)는 2026-09-17에 로컬 저장소에서 삭제했다. 과거 영상 실험을
재현하려면 별도로 보관한 원본 영상 경로를 `--source`에 지정해야 한다.

## 파인튜닝 상태

기존 공개 데이터의 신호등 검출기와 색상 분류기(MobileNetV3-Small)는 학습을 완료했다.
앱 화질 검출기는 별도로 재학습·평가했으며, 새 분할의 학습은 진행 중이고 실제 앱 정답 평가는 남아 있다.
`tools/`의 학습 스크립트는 데이터가 추가되거나 성능을 개선할 때의 재학습·재현 용도로 보존한다.
횡단보도 연결은 별도 실험 로직이므로 실사용 전 다중 신호등 영상 평가가 필요하다.

## 1. 현재 목표

보행자 신호등을 다음 흐름으로 처리한다.

```text
YOLO 객체 검출
  → pedestrian_signal 위치 검출
  → 검출 박스 crop
  → 색상 분류기
  → red / green / unknown
  → 횡단보도↔신호등 후보 연결
```

YOLO는 위치만 담당하고, 색상 분류기는 crop 이미지의 색만 담당한다. 현재 검출기는
`pedestrian_signal`과 `crosswalk`를 동시에 학습한다. `red`, `green`을 YOLO
클래스로 사용하면 별도 색상 분류기와 역할이 중복된다.

## 2. 구현된 파일

### `extract_video_frames.py`

영상 또는 영상 폴더에서 학습·라벨링용 프레임을 추출한다. 추출 위치, 원본 영상,
프레임 번호, timestamp를 `manifest.jsonl`에 기록한다.

```bash
python parts/traffic_light/tools/extract_video_frames.py \
  --source data \
  --output datasets/traffic_light_frames \
  --sample-fps 3
```

### `prepare_classifier_crops.py`

YOLO 형식뿐 아니라 LabelMe JSON의 색상 라벨도 읽어 신호등 부분만 잘라서 분류기용
ImageFolder 데이터셋을 생성한다. 교차로정보 데이터의 `R_Signal`, `G_Signal`은 각각
`red`, `green`으로 자동 매핑하며 폴더 단위 train/val/test 분할을 지원한다.

```text
0 = red
1 = green
2 = unknown
```

입력은 다음 구조를 사용한다.

```text
dataset/images/train/*.jpg
dataset/images/val/*.jpg
dataset/images/test/*.jpg       # 선택
dataset/labels/train/*.txt
dataset/labels/val/*.txt
dataset/labels/test/*.txt       # 선택
```

실행 예:

```bash
python parts/traffic_light/tools/prepare_classifier_crops.py \
  --images datasets/ped_signal_yolo/images \
  --labels datasets/ped_signal_yolo/labels \
  --output datasets/traffic_light_classifier \
  --class-names red green unknown \
  --padding 0.10
```

결과는 `train/<class>`, `val/<class>`, `test/<class>` 구조이며 crop 좌표와 원본은
`manifest.jsonl`에 저장된다. 인접 프레임 누수를 막기 위해 split은 프레임 단위가 아니라
영상 단위로 나눈다.

### `prepare_yolo_signal_dataset.py`

교차로정보 LabelMe JSON의 `R_Signal`, `G_Signal`을 YOLO class 0
`pedestrian_signal`로 합친다. 신호가 없는 사진 일부도 negative image로 포함한다. 기본은
원본 이미지 심볼릭 링크라서 전체 이미지를 중복 복사하지 않는다. `data.yaml`, YOLO txt,
`manifest.jsonl`, `summary.json`을 생성한다.

### `train_signal_detector.py`

생성한 `data.yaml`과 사전학습 `yolo26s.pt`로 단일 클래스 보행자 신호등 검출기를
파인튜닝한다. 최고 가중치는
`runs/traffic_light_detector/<실행시각>/weights/best.pt`에 저장된다.

### `train_signal_classifier.py`

동일한 crop 데이터셋으로 MobileNetV3-Small과 EfficientNet-B0를 각각 파인튜닝한다.
ImageNet 사전학습 가중치를 사용하고, 클래스 불균형을 보정한다. 최고 모델은 validation
accuracy가 아니라 클래스별 recall 평균인 `macro_recall` 기준으로 저장한다.

```bash
python parts/traffic_light/tools/train_signal_classifier.py \
  --data datasets/traffic_light_classifier \
  --model both \
  --epochs 30 \
  --batch-size 64 \
  --device 0 \
  --amp
```

결과 위치:

```text
runs/traffic_light_classifier/<실행ID>/
├── mobilenet_v3_small/best.pt
├── efficientnet_b0/best.pt
└── comparison.json
```

`best.pt`가 실제 추론에 사용하는 파인튜닝 체크포인트다. `history.json`에는 epoch별
학습 기록이, `result.json`에는 recall·confusion matrix·파라미터 수가 저장된다.

### `runtime/pipeline.py`와 `tools/benchmark_yolo_classifier.py`

YOLO와 색상 분류기를 연결하여 실제 이미지·영상에서 처리 속도를 측정한다.

```bash
python parts/traffic_light/tools/benchmark_yolo_classifier.py \
  --source data/test.mp4 \
  --detector weights/yolo_pedestrian_signal.pt \
  --signal-classes pedestrian_signal \
  --classifier-model mobilenet_v3_small \
  --classifier-weights runs/traffic_light_classifier/<실행ID>/mobilenet_v3_small/best.pt \
  --no-save-media
```

검출 crop이 너무 작으면 제외하고, 분류기 confidence가 기본 0.60 미만이면 `unknown`으로
처리한다. 원래 최고 클래스는 `frames.jsonl`의 `raw_class_name`에 기록된다. 여러 crop은
한 프레임에서 배치로 분류하여 불필요한 모델 호출을 줄인다.

## 3. 실제 데이터 준비 결과

원본 `교차로정보 데이터셋_bbox_1`~`bbox_6`의 이미지 36,808장과 같은 이름의 LabelMe
JSON을 읽어 다음 두 데이터셋을 생성했다. 원본 파일은 수정하지 않았다.

### 색상 분류기 데이터

경로: `datasets/intersection_signal_classifier`

```text
전체 crop: 14,522장
red:       10,712장
green:      3,810장
train:     13,594장
val:          808장
test:         120장
```

`Zebra_Cross` 37,110개와 잘못된 `1` 라벨 5개는 제외했다. 공개 데이터에 꺼짐·가려짐
정답이 없어서 현재 분류기는 `red/green` 2클래스로 학습한다. `unknown`은 추후 직접 촬영
데이터로 추가해야 한다.

### YOLO 데이터

경로: `datasets/intersection_pedestrian_signal_yolo`

```text
선택 이미지: 16,906장
신호등 박스: 14,522개
train:       16,302장 / 13,594 boxes
val:            559장 /    808 boxes
test:            45장 /    120 boxes
```

`R_Signal`, `G_Signal`을 class 0 `pedestrian_signal`로 합쳤고 train/val에는 신호등이
없는 배경 사진을 약 20% 포함했다. 이미지는 원본에 대한 심볼릭 링크이며 끊어진 링크는
0개다. 생성 데이터는 `.gitignore`의 `datasets/` 규칙으로 Git에서 제외한다.

## 4. 색상 분류기 파인튜닝 결과

실행 경로: `runs/traffic_light_classifier/20260915T085300Z_3f63cc9d`

MobileNetV3-Small과 EfficientNet-B0를 같은 데이터와 설정으로 학습했다. 명령에는
`--epochs 30`을 지정했지만 validation macro recall이 더 이상 개선되지 않아 조기 종료가
작동했다.

| 항목 | MobileNetV3-Small | EfficientNet-B0 |
|---|---:|---:|
| 실제 실행 epoch | 10 | 9 |
| 최고 checkpoint epoch | 3 | 2 |
| validation accuracy | 99.876% | 99.876% |
| validation macro recall | 99.913% | 99.913% |
| validation 오분류 | 1/808 | 1/808 |
| test accuracy | 100% | 100% |
| test macro recall | 100% | 100% |
| test loss | 0.001497 | 0.000309 |
| 파라미터 수 | 1,519,906 | 4,010,110 |
| checkpoint 크기 | 약 6MB | 약 16MB |
| 학습시간 | 58.7초 | 108.8초 |

최초 라벨 기준으로 두 모델 모두 validation에서 red 572장 중 1장을 green으로 분류했고
green 236장은 모두 맞혔다. test 120장은 모두 맞혔다. 정확도 차이가 없는 상태에서
MobileNet은 파라미터가 약 62% 적고 학습시간도 약 46% 짧으므로 현재 1순위 모델로
선택한다.

EfficientNet의 test loss가 더 낮지만 test가 120장뿐이라 실제 환경 우위를 의미한다고
보기 어렵다. 현재 데이터에는 `unknown`, 역광, 가려짐 및 YOLO 검출 오차가 반영된 crop이
없으므로 실제 영상에서 두 모델의 추론시간과 오분류를 추가 비교해야 한다. `runs/`의
체크포인트와 결과 JSON은 Git에 커밋하지 않는다.

### Validation 라벨 오류 수정

두 모델이 공통으로 green으로 분류했던 다음 원본을 확인했다.

```text
교차로정보 데이터셋_bbox_4/MP_SEL_B002677.jpg
교차로정보 데이터셋_bbox_4/MP_SEL_B002677.json
문제 박스: shape 2, [1274.90, 4.45] ~ [1358.26, 158.62]
```

crop을 육안으로 확인한 결과 초록색 보행 신호였지만 JSON에는 `R_Signal`로 잘못
기록되어 있었다. 이를 `G_Signal`로 수정하고 분류기 데이터셋 전체를 다시 생성했다.
수정 후 validation 클래스 수는 green 237장, red 571장이다.

기존 최고 checkpoint로 수정된 crop을 다시 추론한 결과는 다음과 같다.

| 모델 | 예측 | green 확률 |
|---|---|---:|
| MobileNetV3-Small | green | 99.895% |
| EfficientNet-B0 | green | 99.995% |

나머지 807장은 변경되지 않았고 기존 평가에서 모두 정답이었으므로, 수정된 validation
정답 기준 두 모델의 accuracy와 macro recall은 모두 100%다. 해당 파일은 validation에만
있어서 학습 가중치를 다시 생성할 필요는 없다. 기존 `comparison.json`은 수정 전 라벨로
실행한 이력을 보존한다.

## 5. YOLO 파인튜닝 결과

YOLO는 16,302장의 train 이미지와 13,594개의 `pedestrian_signal` 박스로 학습했다.
먼저 `yolo26s.pt`를 5 epoch 시험 학습한 뒤, 그 실행의 최고 가중치에서 추가 학습했다.

```text
최초 5 epoch:
runs/traffic_light_detector/20260915_최초_5epoch

추가 학습:
runs/traffic_light_detector/20260916_추가학습
```

추가 학습은 25 epoch를 지정했으나 15 epoch에서 최고점을 기록한 뒤 성능이 개선되지 않아
patience 7에 따라 22 epoch에서 조기 종료됐다. 실제 추론에는 `last.pt`가 아니라 다음
가중치를 사용한다.

```text
runs/traffic_light_detector/20260916_추가학습/weights/best.pt
```

동일한 validation split과 `imgsz=960`으로 파인튜닝 전후를 다시 평가한 결과는 다음과 같다.

| 항목 | 파인튜닝 전 | 파인튜닝 후 | 변화 |
|---|---:|---:|---:|
| Precision | 36.66% | 91.36% | +54.70%p |
| Recall | 55.32% | 90.31% | +34.99%p |
| mAP50 | 32.93% | 93.22% | +60.29%p |
| mAP50-95 | 17.63% | 67.60% | +49.97%p |

파인튜닝 후 F1은 약 90.83%다. validation F1-confidence 곡선은 confidence 약 0.378에서
최고점을 보였으므로 실제 영상의 첫 기준은 `--conf 0.38`로 정한다. 미탐이 중요하면
0.25까지 낮추고 오탐과 함께 다시 비교한다.

결과 기록용 핵심 파일은 추가 학습 폴더의 `weights/best.pt`와 `weights/results.csv`다.
`runs/`는 Git에서 제외되므로 체크포인트와 실행 결과는 별도로 보관한다.

## 6. 실제 영상 통합 테스트

최종 후보는 파인튜닝한 단일 클래스 YOLO와 MobileNetV3-Small이다.

```text
YOLO:
runs/traffic_light_detector/20260916_추가학습/weights/best.pt

색상 분류기:
runs/traffic_light_classifier/20260915T085300Z_3f63cc9d/
  mobilenet_v3_small/best.pt
```

기존 학습 데이터와 다른 `신호등1.mp4` 300프레임으로 연결을 확인한 뒤, 별도 촬영 영상
3개를 처음부터 끝까지 처리했다. 공통 설정은 YOLO confidence 0.38, 분류기 confidence
0.70, `imgsz=960`이다.

```bash
.venv/bin/python parts/traffic_light/tools/benchmark_yolo_classifier.py \
  --source "/mnt/c/Users/10/Desktop/2차플젝/yolo 객체 탐지 분류기 모델 속도/신호등 원본 동영상/신호등1.mp4" \
  --detector runs/traffic_light_detector/20260916_추가학습/weights/best.pt \
  --signal-classes pedestrian_signal \
  --classifier-model mobilenet_v3_small \
  --classifier-weights \
    runs/traffic_light_classifier/20260915T085300Z_3f63cc9d/mobilenet_v3_small/best.pt \
  --device 0 \
  --detector-imgsz 960 \
  --conf 0.38 \
  --classifier-min-confidence 0.70 \
  --max-frames 300
```

결과 폴더에는 표시 영상 `result.mp4`, 프레임별 탐지·색상 결과 `frames.jsonl`, 실행 설정
`config.json`, 속도 요약 `summary.json`이 저장된다. 실제 정확도 평가는 영상의 정답 라벨이
없으므로 결과 영상을 육안 확인하고 오탐·미탐·색상 오분류 프레임을 별도로 기록해야 한다.

### 실제 영상 실행 결과

| 영상 | 처리 프레임 | 탐지 수 | red | green | unknown | 평균 처리시간 | 예상 FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 신호등1 일부 | 300 | 161 | 151 | 0 | 10 | 17.83ms | 56.09 |
| B_05 전체 | 3,236 | 3,267 | 2,084 | 1,178 | 5 | 17.11ms | 58.45 |
| C_09 전체 | 1,592 | 1,538 | 876 | 659 | 3 | 19.89ms | 50.27 |
| C_10 전체 | 6,564 | 6,796 | 4,588 | 2,181 | 27 | 22.39ms | 44.66 |

네 실행 모두 파인튜닝한 YOLO와 MobileNet 가중치를 정상적으로 불러왔고 GPU에서 완료됐다.
전체 영상 3개의 측정 처리속도는 약 44.66~58.45 FPS였다. 이 수치는 파일 읽기와 결과 영상
저장을 포함한 현재 PC의 실행 결과이며, 영상별 해상도와 검출 수에 따라 달라진다.

첫 시험에서 YOLO가 신호등과 형태가 비슷한 도로 볼라드를 검출했고, 해당 crop의 MobileNet
confidence는 최대 66.24%였다. 같은 구간의 실제 빨간 신호등은 최소 92.98%였으므로 색상을
잘못 확정할 가능성을 줄이기 위해 `--classifier-min-confidence 0.70`을 적용했다. 이는 한
영상에서 정한 초기 운영값이며 더 다양한 환경에서 다시 조정해야 한다. YOLO 오탐 자체가
제거된 것은 아니므로 `unknown`도 안전한 신호로 간주해서는 안 된다.

위 표의 red/green/unknown은 모델 출력 횟수이며 정답 개수가 아니다. 영상에 프레임별 정답
라벨이 없기 때문에 이 결과만으로 실제 영상 정확도를 계산할 수 없다.

## 7. 아직 필요한 작업

1. 전체 결과 영상에서 YOLO 미탐·오탐과 red/green 오분류 구간 기록
2. 대표 프레임에 정답을 붙여 실영상 detector와 classifier 정확도 계산
3. 볼라드 등 반복 오탐 이미지를 YOLO negative 데이터로 보강
4. 직접 촬영 데이터에서 `unknown`(꺼짐·가려짐·판독 불가) 라벨 추가
5. 실제 서비스 연결 전에 여러 프레임 연속 확인 및 음성 발화 로직 추가

## 8. 검증 상태

이전 단계에서는 테스트 29개가 통과했다. 분류기 GPU 학습과 validation/test 평가까지 완료했다.
YOLO 파인튜닝과 validation 평가를 완료했고, 최종 MobileNet과 YOLO를 연결한 실제 영상
4건의 실행 및 속도 측정도 완료했다. 정답 라벨 기반 실영상 정확도 평가는 아직 필요하다.
