# 신호등 파트 진행 문서

## 횡단보도 연결 실험 (2026-09-17)

2클래스 YOLO의 `crosswalk` 검출을 `benchmark_yolo_classifier.py`에서 사용할 수 있도록
`--associate-crosswalk` 옵션을 추가했다. 신호등이 1개면 그 신호등만 분류해
`single_signal`로 기록하고, 횡단보도와의 관계는 미확인으로 남긴다. 2개 이상이면 화면
아래·중앙의 횡단보도, 선분 교차로 추정한 소실점, 신호등의 방향·크기, 영상 프레임 간
같은 박스 유지 여부로 후보를 고른다. 선택된 신호등 하나만 색상 분류·표시한다.
`frames.jsonl`에 `association`과 `crosswalks`를 기록한다. 소실점 추정 실패·후보 모호성·
시간적 불일치는 `unknown`이다. 실제 테스트 사진 1장에서는 횡단보도 2개, 신호등 0개가
검출되어 신호등을 연결할 수 없었다. 연결 정확도는 아직 평가되지 않았다.

실행 예:

```bash
.venv/bin/python parts/traffic_light/benchmark_yolo_classifier.py \
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
.venv/bin/python parts/traffic_light/benchmark_yolo_classifier.py \
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

현재 테스트 29개가 모두 통과한다. 분류기 GPU 학습과 validation/test 평가까지 완료했다.
YOLO 파인튜닝과 validation 평가를 완료했고, 최종 MobileNet과 YOLO를 연결한 실제 영상
4건의 실행 및 속도 측정도 완료했다. 정답 라벨 기반 실영상 정확도 평가는 아직 필요하다.
