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
red:       10,713장
green:      3,809장
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

## 4. 아직 필요한 작업

1. 분류기와 YOLO를 각각 5 epoch 시험 학습
2. YOLO 파인튜닝 후 recall과 차량 신호등 오검출 확인
3. MobileNetV3-Small과 EfficientNet-B0를 같은 split으로 30 epoch 비교
4. 직접 촬영 데이터에서 `unknown`(꺼짐·가려짐·판독 불가) 라벨 추가
5. 공개 데이터와 겹치지 않는 실제 영상으로 최종 평가
6. 실제 서비스 연결 전에 여러 프레임 연속 확인 및 음성 발화 로직 추가

## 5. 검증 상태

현재 테스트 32개가 모두 통과한다. 현재 실행 환경에서는 CUDA가 비활성 상태라 실제 GPU
학습과 실제 데이터 정확도는 CUDA가 연결된 환경에서 별도로 측정해야 한다.
