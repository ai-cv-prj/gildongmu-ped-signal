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

YOLO 형식의 색상 라벨을 읽어 신호등 부분만 잘라서 분류기용 ImageFolder 데이터셋을
생성한다. 기본 class id는 다음과 같다.

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

## 3. 아직 필요한 작업

1. 직접 촬영한 영상을 영상 단위로 train/val/test 분리
2. 보행자 신호등 bounding box 라벨링
3. YOLO 검출용 라벨은 `pedestrian_signal` 단일 클래스로 준비
4. 분류기용 라벨은 `red/green/unknown`으로 준비
5. YOLO 파인튜닝 후 recall과 차량 신호등 오검출 확인
6. 두 분류기를 같은 영상에서 실행하여 정확도와 지연 비교
7. 실제 서비스 연결 전에 여러 프레임 연속 확인 및 음성 발화 로직 추가

## 4. 검증 상태

현재 테스트 22개가 모두 통과한다. 실제 GPU 영상 추론과 실제 데이터 정확도는 데이터와
YOLO/분류기 가중치를 준비한 뒤 별도로 측정해야 한다.
