# YOLO 검출 → 신호등 색 판별 테스트

이 도구는 한 프레임을 다음 순서로 처리하고 각 단계의 시간을 기록합니다.

```text
원본 프레임
  → YOLO26s: 신호등 박스 검출
  → 박스에 여백 추가 및 crop
  → HSV 규칙 기반 또는 학습된 MobileNetV3-Small/EfficientNet-B0로 빨강/초록/구분 불가 판별
```

기본 `--color-method hsv`는 별도 학습 없이 바로 색을 시험할 수 있습니다. YOLO 기본
가중치는 COCO의 일반 `traffic light` 위치를 사용합니다. 작은 신호등, 역광, 색 번짐이
심한 환경에서 정확도를 높이려면 라벨 데이터로 신경망 분류기를 학습해 사용하세요.
COCO `traffic light`는 차량 신호등도 포함하므로 최종 시스템에서는
`pedestrian_signal` 단일 클래스로 파인튜닝한 YOLO 가중치를 권장합니다.

## 학습 없이 바로 색 구분 테스트

CPU에서 영상 일부를 먼저 확인합니다.

```bash
python parts/traffic_light/benchmark_yolo_classifier.py \
  --source data/신호등1.mp4 \
  --device cpu \
  --color-method hsv \
  --vid-stride 10 \
  --max-frames 30
```

결과 영상의 박스 위에는 `red`, `green`, `unknown` 세 결과만 표시됩니다. 노란색이나
다른 색은 `unknown`(구분 불가)으로 처리합니다. 색 픽셀이 너무 적어 `unknown`이 자주 나오면
`--hsv-min-color-ratio 0.005`처럼 낮춰 비교할 수 있습니다. 반대로 주변 간판 색을
잘못 읽으면 값을 높이세요. 원시 색 비율은 `frames.jsonl`의 `color_ratios`에 남습니다.

## 설치

팀 Python 3.14.4 가상환경과 CUDA 12.8 PyTorch를 먼저 준비합니다.

```bash
cd ~/ai_cv_prj
source .venv/bin/activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r parts/traffic_light/requirements.txt
python scripts/check_gpu.py
```

## 라벨링용 영상 프레임 추출

`data/`의 모든 영상을 초당 3장씩 추출합니다. 영상마다 별도 폴더가 생성되며,
`manifest.jsonl`에는 원본 영상·프레임 번호·시간 정보가 기록됩니다.

```bash
python parts/traffic_light/extract_video_frames.py \
  --source data \
  --output datasets/traffic_light_frames \
  --sample-fps 3
```

먼저 영상별 10장만 시험하려면 다음처럼 실행합니다.

```bash
python parts/traffic_light/extract_video_frames.py \
  --source data \
  --output datasets/traffic_light_frames_preview \
  --sample-fps 3 \
  --max-frames-per-video 10
```

기존 파일을 실수로 섞거나 덮어쓰지 않도록 출력 폴더가 비어 있지 않으면 중단합니다.
다시 추출할 때는 새로운 `--output` 경로를 지정하세요.

## 분류기 학습용 crop 만들기

분류기는 YOLO가 찾은 작은 박스만 입력받습니다. 학습 데이터도 원본 전체 사진이 아니라
정답 박스를 잘라 만든 crop이어야 합니다. 먼저 영상 단위로 `train/val/test`를 나눈 뒤,
각 사진에서 보행자 신호등 박스와 색을 YOLO 형식으로 라벨링합니다. 인접 영상 프레임을
무작위로 나누면 거의 같은 장면이 train과 val에 동시에 들어가 정확도가 부풀려집니다.

입력 구조와 class id는 다음을 사용합니다.

```text
datasets/ped_signal_yolo/
├── images/train/*.jpg       # 원본 영상 기준 약 70%
├── images/val/*.jpg         # 원본 영상 기준 약 20%
├── images/test/*.jpg        # 원본 영상 기준 약 10% (선택)
└── labels/{train,val,test}/*.txt

class 0 = red, class 1 = green, class 2 = unknown
```

여기서 `unknown`은 꺼짐, 노란불, 가려짐, 판독 불가를 뜻합니다. 자동차 신호등은
분류기에 넣기 전에 YOLO 검출 라벨에서 제외하거나 별도 negative 정책을 정해야 합니다.
이 색 라벨은 crop 생성용 정답입니다. 별도로 YOLO 검출기를 파인튜닝할 때는 세 class id를
모두 `pedestrian_signal` 하나로 합친 복사본을 사용하세요. 그래야 YOLO는 위치, 분류기는
색이라는 역할이 유지됩니다.

정답 박스에 10% 여백을 더한 뒤 ImageFolder 구조로 crop합니다.
라벨 한 줄은 `class_id center_x center_y width height`의 YOLO bbox 5열 형식이어야 합니다.

```bash
python parts/traffic_light/prepare_classifier_crops.py \
  --images datasets/ped_signal_yolo/images \
  --labels datasets/ped_signal_yolo/labels \
  --output datasets/traffic_light_classifier \
  --class-names red green unknown \
  --padding 0.10
```

결과는 `train/red`, `train/green`, `train/unknown`과 같은 구조이며 `manifest.jsonl`에
모든 crop의 원본 파일과 좌표가 남습니다. 원본 라벨이 단일 `pedestrian_signal` 클래스뿐이면
색 정답이 없으므로 이 단계 전에 red/green/unknown 라벨을 추가해야 합니다.

## MobileNetV3-Small / EfficientNet-B0 학습 비교

두 후보를 동일한 split과 설정으로 차례로 학습합니다. 작은 데이터셋이므로 ImageNet
사전학습 백본을 기본으로 사용하고, 클래스 수 불균형도 loss에서 보정합니다.

```bash
python parts/traffic_light/train_signal_classifier.py \
  --data datasets/traffic_light_classifier \
  --model both \
  --epochs 30 \
  --batch-size 64 \
  --device 0 \
  --amp
```

각 실행은 `runs/traffic_light_classifier/<실행ID>/`에 생성됩니다.

- `mobilenet_v3_small/best.pt`, `efficientnet_b0/best.pt`: 최고 val accuracy 체크포인트
- 각 모델의 `history.json`, `result.json`: epoch별/최종 결과
- `comparison.json`: 두 모델의 accuracy, macro recall, confusion matrix, 파라미터 수와 학습시간 비교

불균형 데이터에서 다수 클래스만 잘 맞히는 모델을 피하려고 최고 체크포인트는 val
`macro_recall` 기준으로 고릅니다. 정확도만으로 고르지 말고 클래스별 recall과 아래의 실제
영상 벤치마크에서 `total_pipeline_wall`도 함께
비교하세요. 0.5초 전체 지연 목표에서는 보통 MobileNetV3-Small을 먼저 기준 모델로 삼고,
EfficientNet-B0의 정확도 상승이 실제로 의미 있을 때만 교체하는 편이 안전합니다.

학습된 MobileNet 체크포인트를 전체 파이프라인에 연결하는 예시는 다음과 같습니다.

```bash
python parts/traffic_light/benchmark_yolo_classifier.py \
  --source data/walk.mp4 \
  --detector weights/yolo_pedestrian_signal_best.pt \
  --signal-classes pedestrian_signal \
  --color-method neural \
  --classifier-model mobilenet_v3_small \
  --classifier-weights runs/traffic_light_classifier/<실행ID>/mobilenet_v3_small/best.pt \
  --no-save-media
```

EfficientNet은 두 `mobilenet_v3_small` 부분을 `efficientnet_b0`로 바꿔 같은 영상과 옵션으로
실행합니다. 두 실행의 `summary.json`에서 정확도가 아니라
`mean_total_pipeline_wall_ms`, `estimated_pipeline_fps`, `peak_cuda_memory_mb`를 비교합니다.
신경망 최고 확률이 기본 0.60보다 낮으면 오발화를 줄이기 위해 `unknown`으로 바뀌며,
원래 최고 클래스는 `frames.jsonl`의 `raw_class_name`에 남습니다. 검출 crop의 가로 또는
세로가 6픽셀보다 작아도 색 판별을 건너뜁니다. 각각
`--classifier-min-confidence`, `--min-crop-size`로 조정할 수 있습니다.

## 교차로정보 LabelMe 데이터로 파인튜닝

현재 확인한 원본 폴더는 이미지와 같은 이름의 LabelMe JSON을 함께 가지고 있습니다.
JSON의 `R_Signal`과 `G_Signal` 박스를 그대로 이용하므로 이 데이터는 다시 직접 라벨링할
필요가 없습니다. `Zebra_Cross`와 잘못 들어간 `1` 라벨은 자동으로 제외합니다.

아래 명령은 프로젝트 가상환경이 활성화된 상태를 기준으로 합니다. 터미널을 새로 열었다면
먼저 다음 명령을 실행하고 프롬프트 앞에 `(.venv)`가 표시되는지 확인합니다.

```bash
cd ~/ai_cv_prj
source .venv/bin/activate
```

우선 아래처럼 폴더 단위로 나눕니다. 이는 첫 실험용 분할이며, 최종 성능은 이 데이터와
겹치지 않는 직접 촬영 영상으로 다시 확인해야 합니다.

```text
bbox_1, bbox_2 → train
bbox_3, bbox_4 → val
bbox_5, bbox_6 → test
```

### 1. 색 분류기용 crop 생성

원본 LabelMe 박스에서 신호등 부분만 잘라 `red/green` 두 클래스 ImageFolder를 만듭니다.
이 공개 데이터에는 꺼짐·가려짐을 뜻하는 `unknown` 정답이 없으므로 첫 모델은 2클래스로
학습하고, 이후 직접 촬영 데이터로 `unknown`을 추가합니다.

```bash
python parts/traffic_light/prepare_classifier_crops.py \
  --images "/mnt/c/Users/10/Desktop/2차플젝/(2차_최종) 교차로정보 데이터셋_20210720" \
  --label-format labelme \
  --output datasets/intersection_signal_classifier \
  --class-names red green \
  --label-map R_Signal=red G_Signal=green \
  --split-map \
    "교차로정보 데이터셋_bbox_1=train" \
    "교차로정보 데이터셋_bbox_2=train" \
    "교차로정보 데이터셋_bbox_3=val" \
    "교차로정보 데이터셋_bbox_4=val" \
    "교차로정보 데이터셋_bbox_5=test" \
    "교차로정보 데이터셋_bbox_6=test" \
  --padding 0.10
```

생성이 끝나면 `datasets/intersection_signal_classifier/summary.json`에서 red/green 수와
train/val/test 수를 먼저 확인합니다. 출력 폴더가 비어 있지 않으면 안전을 위해 중단하므로
재실행할 때는 새 출력 경로를 사용하세요.

### 2. MobileNet과 EfficientNet 파인튜닝

처음에는 MobileNet 하나를 5 epoch만 학습해 전체 과정이 정상인지 확인합니다.

```bash
python parts/traffic_light/train_signal_classifier.py \
  --data datasets/intersection_signal_classifier \
  --model mobilenet_v3_small \
  --epochs 5 \
  --batch-size 64 \
  --device 0 \
  --amp
```

정상 학습이 확인되면 두 후보를 30 epoch까지 비교합니다.

```bash
python parts/traffic_light/train_signal_classifier.py \
  --data datasets/intersection_signal_classifier \
  --model both \
  --epochs 30 \
  --batch-size 64 \
  --device 0 \
  --amp
```

파인튜닝된 분류기 가중치는
`runs/traffic_light_classifier/<실행ID>/<모델명>/best.pt`에 저장됩니다.

### 3. YOLO용 단일 클래스 데이터 생성

YOLO는 빨강/초록을 구별하지 않고 둘 다 class 0 `pedestrian_signal`로 합칩니다. 신호등이
없는 사진도 신호등 사진 수의 20%만큼 넣어 배경 오검출을 줄입니다. 기본 `symlink` 방식은
원본 사진을 복사하지 않아 저장 공간을 아끼며, WSL 안에서 학습할 때 그대로 사용할 수 있습니다.

```bash
python parts/traffic_light/prepare_yolo_signal_dataset.py \
  --source "/mnt/c/Users/10/Desktop/2차플젝/(2차_최종) 교차로정보 데이터셋_20210720" \
  --output datasets/intersection_pedestrian_signal_yolo \
  --split-map \
    "교차로정보 데이터셋_bbox_1=train" \
    "교차로정보 데이터셋_bbox_2=train" \
    "교차로정보 데이터셋_bbox_3=val" \
    "교차로정보 데이터셋_bbox_4=val" \
    "교차로정보 데이터셋_bbox_5=test" \
    "교차로정보 데이터셋_bbox_6=test" \
  --negative-ratio 0.20
```

다른 PC로 데이터셋 폴더 자체를 옮길 계획이면 `--image-mode copy`를 추가합니다. 링크 방식은
원본 폴더를 이동하거나 삭제하면 끊어집니다.

### 4. YOLO 파인튜닝

먼저 5 epoch 시험 학습을 실행합니다.

```bash
python parts/traffic_light/train_signal_detector.py \
  --data datasets/intersection_pedestrian_signal_yolo/data.yaml \
  --model yolo26s.pt \
  --epochs 5 \
  --imgsz 960 \
  --batch 16 \
  --device 0
```

GPU 메모리 부족이 나오면 `--batch 8`, 그래도 부족하면 `--imgsz 640`으로 낮춥니다. 시험이
정상이라면 `--epochs 30`으로 본 학습합니다. 파인튜닝된 YOLO 가중치는
`runs/traffic_light_detector/<실행시각>/weights/best.pt`에 저장됩니다.

### 5. 두 파인튜닝 모델 연결

```bash
python parts/traffic_light/benchmark_yolo_classifier.py \
  --source data/test.mp4 \
  --detector runs/traffic_light_detector/<실행시각>/weights/best.pt \
  --signal-classes pedestrian_signal \
  --color-method neural \
  --classifier-model mobilenet_v3_small \
  --classifier-weights runs/traffic_light_classifier/<실행ID>/mobilenet_v3_small/best.pt
```

분류기 체크포인트 안에 `class_names: [green, red]`처럼 실제 폴더 정렬 순서가 함께 저장되므로
추론 시 클래스 순서를 따로 추측할 필요는 없습니다.

## 처리 속도 측정

사진:

```bash
python parts/traffic_light/benchmark_yolo_classifier.py \
  --source data/signal.jpg --color-method hsv
```

영상 100개 처리 프레임:

```bash
python parts/traffic_light/benchmark_yolo_classifier.py \
  --source data/walk.mp4 \
  --color-method hsv \
  --vid-stride 3 \
  --max-frames 100 \
  --no-save-media
```

`--vid-stride 3`은 30fps 원본의 세 프레임마다 하나를 선택합니다. 실시간 10fps를
보장하는 옵션은 아닙니다. 순수 지연 측정에는 파일 저장과 화면 표시를 끄는
`--no-save-media`를 사용하세요.

신호등이 한 번도 검출되지 않으면 색 판별은 실행되지 않으므로 해당 시간은 0ms입니다.

## 학습된 색 분류기로 측정

### ImageNet 기본 가중치를 넣은 상태로 속도 확인

아래 명령은 ImageNet 사전학습 백본을 내려받아 실제 가중치가 들어간 모델의 속도를
측정합니다. 마지막 `red / green / unknown` 출력층은 아직 학습되지 않았으므로 색 결과는
`untrained_timing_only`로 기록됩니다.

```bash
python parts/traffic_light/benchmark_yolo_classifier.py \
  --source data/신호등1.mp4 \
  --device cpu \
  --color-method neural \
  --classifier-model mobilenet_v3_small \
  --imagenet-pretrained \
  --vid-stride 30 \
  --max-frames 10 \
  --no-save-media
```

EfficientNet-B0를 확인하려면 `--classifier-model efficientnet_b0`로 바꾸면 됩니다. 최초
실행에는 torchvision 가중치 다운로드를 위한 인터넷 연결이 필요합니다.

정확도와 무관하게 `red/green/unknown` 출력까지 이어지는지만 확인하려면 다음 옵션을
추가할 수 있습니다.

```bash
python parts/traffic_light/benchmark_yolo_classifier.py \
  --source data/신호등1.mp4 \
  --device cpu \
  --color-method neural \
  --classifier-model mobilenet_v3_small \
  --imagenet-pretrained \
  --allow-untrained-predictions \
  --vid-stride 10 \
  --max-frames 30
```

화면에는 `UNTRAINED red`처럼 표시됩니다. ImageNet 백본만 사전학습되었고 3클래스
출력층은 임의 가중치이므로, 이 결과는 정확도나 색 판별 가능성을 평가하는 데 사용하면
안 됩니다. 오직 YOLO → crop → 신경망 → 클래스 표시 연결 시험용입니다.

### 신호등 색으로 학습된 가중치 사용

체크포인트는 다음 형식을 권장합니다.

```python
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_names": ["red", "green", "unknown"],
    },
    "weights/efficientnet_b0_signal.pt",
)
```

실행:

```bash
python parts/traffic_light/benchmark_yolo_classifier.py \
  --source data/walk.mp4 \
  --color-method neural \
  --classifier-model efficientnet_b0 \
  --classifier-weights weights/efficientnet_b0_signal.pt \
  --vid-stride 3 \
  --max-frames 300 \
  --no-save-media
```

체크포인트에 `class_names`가 없으면 기본 순서 `red green unknown`을 사용합니다.
학습 당시 출력 노드 순서와 반드시 같아야 합니다. 다른 순서라면 명시합니다.

```bash
python parts/traffic_light/benchmark_yolo_classifier.py \
  --source data/walk.mp4 \
  --color-method neural \
  --classifier-model efficientnet_b0 \
  --classifier-weights weights/efficientnet_b0_signal.pt \
  --class-names green red unknown
```

YOLO도 보행자 신호등 위치에 파인튜닝했다면 바꿔서 실행할 수 있습니다.

```bash
python parts/traffic_light/benchmark_yolo_classifier.py \
  --source data/walk.mp4 \
  --detector weights/yolo_signal_best.pt \
  --color-method neural \
  --classifier-model efficientnet_b0 \
  --classifier-weights weights/efficientnet_b0_signal.pt
```

FP16 실행:

```bash
python parts/traffic_light/benchmark_yolo_classifier.py \
  --source data/walk.mp4 --color-method neural \
  --classifier-weights weights/efficientnet_b0_signal.pt \
  --max-frames 300 --half --no-save-media
```

## 측정 결과

실행마다 `runs/yolo_classifier_benchmark/<UTC시각_고유ID>/`에 저장됩니다.

- `summary.json`: 평균 단계별 시간, 전체 파이프라인 시간, 추정 FPS, 최대 CUDA 메모리
- `frames.jsonl`: 프레임별 박스, 분류 결과, crop 배치 크기와 단계별 시간
- `config.json`: 실제 옵션, 패키지 버전, 클래스 순서와 가중치 사용 여부
- `result.mp4` 또는 `*_result.jpg`: 박스와 분류 결과가 표시된 미디어

주요 항목:

| 항목 | 포함 범위 |
|---|---|
| `detector_pipeline_wall` | YOLO 전처리·추론·후처리 |
| `crop_preprocess` | 박스 확장·crop (신경망 모드는 224×224 패딩·정규화·배치 생성 포함) |
| `classifier_input_transfer` | 분류기 배치를 CPU에서 선택한 장치로 전송 |
| `classifier_inference` | 한 프레임의 HSV 판별 전체 또는 신경망 배치 forward |
| `classifier_postprocess` | softmax와 최고 클래스 선택 |
| `total_pipeline_wall` | YOLO 시작부터 분류 결과 생성까지 |

기본 워밍업 3회는 통계에서 제외됩니다. `estimated_pipeline_fps`는 로컬 모델
파이프라인 시간으로 계산하며 폰 JPEG 인코딩, WebSocket 전송, 서버 큐, TTS 시간은
포함하지 않습니다. 최종 0.5초 목표는 통합 서버에서 별도로 측정해야 합니다.

신경망 분류기는 한 프레임에서 검출된 신호등을 한 배치로 처리합니다. 따라서
신경망 모드의 `classifier_inference`는 crop 한 장이 아니라 해당 프레임 배치 전체 시간입니다.
`mean_classifier_inference_ms_when_run`은 실제로 crop이 있었던 프레임만 평균냅니다.

출력 색 예측은 프레임별 원시 결과입니다. 같은 신호등 추적, 건널 방향 선택,
N프레임 확정과 음성 안내는 수행하지 않습니다.

결과 미디어에는 얼굴·번호판 블러가 없습니다. 공유 전에 별도로 처리하고 원본·가중치·결과를
Git에 커밋하지 마세요.

## 코드 검증

```bash
python -m unittest discover -s parts/traffic_light/tests -v
```

작성 환경에서는 GPU 추론을 실행할 수 없어 실제 처리시간은 측정하지 못했습니다.
팀 RTX 5080 환경에서 `scripts/check_gpu.py`를 먼저 통과시킨 뒤 실행하세요.
