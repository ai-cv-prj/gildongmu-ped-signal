# 신호등 파트 진행 문서

## 1. 현재 목표

보행자 신호등을 다음 2단계로 처리한다.

```text
YOLO 객체 검출
  → pedestrian_signal 위치 검출
  → 검출 박스 crop
  → 색상 분류기
  → red / green / unknown
```

YOLO는 위치만 담당하고, 색상 분류기는 crop 이미지의 색만 담당한다. 따라서 YOLO를
파인튜닝할 때는 `pedestrian_signal` 단일 클래스를 권장한다. `red`, `green`을 YOLO
클래스로 사용하면 별도 색상 분류기와 역할이 중복된다.

## 2. 구현된 파일

### `extract_video_frames.py`

영상 또는 영상 폴더에서 학습·라벨링용 프레임을 추출한다. 추출 위치, 원본 영상,
프레임 번호, timestamp를 `manifest.jsonl`에 기록한다.

```bash
python parts/traffic_light/extract_video_frames.py \
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
python parts/traffic_light/prepare_classifier_crops.py \
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
python parts/traffic_light/train_signal_classifier.py \
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

### `benchmark_yolo_classifier.py`

YOLO와 색상 분류기를 연결하여 실제 이미지·영상에서 처리 속도를 측정한다.

```bash
python parts/traffic_light/benchmark_yolo_classifier.py \
  --source data/test.mp4 \
  --detector weights/yolo_pedestrian_signal.pt \
  --signal-classes pedestrian_signal \
  --color-method neural \
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

## 5. 아직 필요한 작업

1. YOLO를 5 epoch 시험 학습한 뒤 본 파인튜닝
2. YOLO recall과 차량 신호등 오검출 확인
3. 두 분류기 checkpoint를 같은 실제 영상에서 속도·오분류 비교
4. 직접 촬영 데이터에서 `unknown`(꺼짐·가려짐·판독 불가) 라벨 추가
5. 공개 데이터와 겹치지 않는 실제 영상으로 최종 평가
6. 실제 서비스 연결 전에 여러 프레임 연속 확인 및 음성 발화 로직 추가

## 6. 검증 상태

현재 테스트 32개가 모두 통과한다. 분류기 GPU 학습과 validation/test 평가까지 완료했다.
YOLO 파인튜닝과 실제 영상의 전체 파이프라인 평가는 아직 필요하다.
