# 앱 전송 화질 파인튜닝 실험 실행 안내

이 문서는 기존 분할을 유지한 앱 화질 실험의 재현 안내다. 2026-09-19 기준 전처리와
검출기 학습·공개 데이터 평가를 완료했다. 앱 화질 색상 분류기 학습 및 정답 라벨 기반
실제 앱 holdout 평가는 실행하지 않았다. 현재 다음 작업인 새 분할 학습은
[APP_QUALITY_RESPLIT.md](APP_QUALITY_RESPLIT.md)를 따른다.
완료한 경로에 아래 명령을 다시 실행하지 않는다. 모든 명령의 작업 디렉터리는 저장소 루트다.

## 학습과 비교 대상

앱 전송 과정의 축소·압축에 맞추기 위한 검출기 학습과 선택적인 색상 분류기 학습을 제공한다.
현재 완료한 것은 검출기 학습이다. 원본 화질 재학습은 제외하며 기존 정답 라벨을 재사용한다.

| 평가 이름 | 검출기 | 색상 분류기 |
|---|---|---|
| `historical` | 보존된 기존 2클래스 best.pt | 보존된 기존 MobileNet best.pt |
| `app` | 원본 YOLO26s에서 canvas 전송 화질로 새 학습 | 원본 ImageNet MobileNet에서 전체 이미지 전송 처리 후 crop으로 새 학습 |

기존 2클래스 모델은 `yolo26s.pt → 신호등 학습 → 추가 신호등 학습 → 횡단보도 추가 학습`을
거쳤다. 보존 경로는 `runs/traffic_light_crosswalk_detector/횡단보도 추가 학습/weights/best.pt`다.
문서에 적혀 있던 `20260916_173303` 폴더 이름과 현재 파일시스템의 이름이 다르다.
`runs/traffic_light_joint_from_yolo26s/`의 실행 설정은 현재 확인되지 않아 기준으로 사용하지 않았다.

기존 모델을 그대로 보존하고 같은 앱 전송 이미지에서 `app - historical` 성능 차이를 확인한다.
이 비교에는 기존 모델과 새 모델의 학습 이력 차이도 포함된다.
새 검출기는 마지막 2클래스 실행의 전체 `args.yaml`을 JSON으로 보존해 재사용한다.
모델 시작점만 원본 `yolo26s.pt`로 바꾸고 `resume=False`를 강제한다. epochs=30, imgsz=960,
batch=16, seed=42, patience=7, lr0=0.01, optimizer=auto, 증강·AMP 등은 동일하다.
`imgsz`는 바꾸지 않으며 별도 전처리된 JPEG를 실제 학습 입력으로 사용한다.
조기 종료 정책은 유지하며 실제 종료 epoch와 선택된 best epoch는 학습 결과에 따라 결정된다.

기존 분류기는 전체 CLI 설정을 저장하지 않았다. 문서의 명령과 기존 코드 기본값으로
MobileNetV3-Small, 30 epochs, batch=64, imgsz=224, AdamW lr=0.001, weight_decay=0.0001,
seed=42, patience=7, class weights, AMP 설정을 복원했다. 과거 설정과 완전히 같다고
검증할 수는 없다. 새 분류기는 이 설정과 SHA256으로 고정한 원본 ImageNet 파일을 사용한다.
새 crop은 전체 이미지의 canvas JPEG를 디코딩한 뒤 PNG로 저장해 추가 압축 손실을 피한다.

## 필요한 파일과 패키지

- 설정: `parts/traffic_light/configs/app_quality.json`
- 기존 검출기 설정 사본: `parts/traffic_light/configs/app_quality_detector_reference.json`
- 검출 데이터: `datasets/intersection_signal_crosswalk_yolo/{manifest.jsonl,data.yaml,images,labels}`
- 분류기 데이터: `datasets/intersection_signal_classifier/manifest.jsonl` (기존 crop의 class/split/xyxy 기록)
- 각 manifest가 가리키는 원본 이미지. 현재 경로는 `/mnt/c/Users/10/Desktop/2차플젝/...`다.
- 원본 가중치: `yolo26s.pt`, `.torch-cache/hub/checkpoints/mobilenet_v3_small-047dcff4.pth`
- 기존 비교 모델: 설정의 `historical_weights` 두 경로
- 학습에 쓰지 않은 실제 앱 세션의 전송 JPEG와 전체 객체 정답 라벨

설정의 가중치 SHA256은 현재 파일에서 기록했다. 새 학습은 가중치 파일이 없거나 해시가
다르면 중단하며 시작 가중치를 다른 파일로 자동 대체하지 않는다. 다른 위치로 옮겼다면 경로만
변경한다. 기존 파인튜닝 모델을 `initial_weights`로 지정하지 않는다.

현재 팀 환경은 Python 3.14, torch 2.11.0+cu128, torchvision 0.26.0+cu128이다.
이미 준비된 `.venv`를 우선 사용한다. 학습·평가 코드는 설정에 기록된 패키지 버전을 확인한다.

```bash
cd /home/user/ai_cv_prj
source .venv/bin/activate
# torch/torchvision이 없는 새 환경일 때만 먼저 설치
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r parts/traffic_light/requirements-app-quality.txt
python -m playwright install chromium
```

Linux 브라우저 시스템 라이브러리가 부족한 경우 `python -m playwright install-deps chromium`을
추가 실행한다. Playwright 설치 버전과 브라우저 바이너리는 묶어서 보관하고 실험 사이에
업데이트하지 않는다. 실험 환경 기록 예:

```bash
mkdir -p runs/app_quality_960_q08_v1
python -m pip freeze > runs/app_quality_960_q08_v1/environment.txt
```

## 1. 설정 검토와 소규모 검사

`app_quality.json`의 경로와 평가 manifest 경로를 먼저 맞춘다. 재실험에서는
`prepared_root`, `runs_root`, `evaluation.output`을 함께 새 경로로 바꾼다.
전처리 이후 설정을 바꾸면 학습이 중단되므로 평가 세션을 넣을 경로도 미리 정한다.

```bash
python -m parts.traffic_light.tools.train_app_quality --task detector --arm app --print-only
python -m unittest parts.traffic_light.tests.test_app_quality -v
```

`--print-only`는 데이터나 모델을 로드하지 않는다. 테스트는 임시 폴더의 합성 이미지와
가짜 모델/수치만 사용한다. Playwright가 설치되어 있으면 실제 브라우저로 합성 이미지
2장을 인코딩하는 smoke test도 수행한다. 모델 학습·추론은 하지 않는다.

## 2. 전체 전처리 (사용자가 실행)

```bash
python -m parts.traffic_light.tools.prepare_app_quality
```

기존 manifest에서 이미지 목록, 배경 음성 샘플, split, 클래스, crop 영역을 읽는다.
새 무작위 분할이나 negative sampling은 하지 않는다. 전체 이미지에 다음 처리를 적용한다.

1. 실제 Chromium의 `Image.decode()`로 원본 디코딩
2. `scale = Math.min(1, 960 / Math.max(width, height))`, 각 크기는 `Math.round`
3. 앱과 같은 기본 canvas 옵션으로 `drawImage`
4. `canvas.toBlob(..., 'image/jpeg', 0.8)`로 저장
5. OpenCV로 JPEG 디코딩 검증 후 검출 학습에 사용; 분류기는 이 디코딩 결과에서 crop

원본 파일은 경로로 브라우저에 전달한다. Python에서 원본 전체를 읽어 Base64 문자열로
복사하지 않는다. 이미지마다 object URL·canvas·파일 입력을 해제하고, 기본 50장마다
Chromium과 Playwright 드라이버를 종료·재시작해 장시간 처리 시 메모리 누적을 제한한다.
이는 960px/quality=0.8 인코딩 조건을 바꾸지 않는다.
[파일 경로 업로드](https://playwright.dev/python/docs/input#upload-files),
[object URL 해제](https://developer.mozilla.org/en-US/docs/Web/API/URL/revokeObjectURL_static).

`OSError: [Errno 12] Cannot allocate memory`로 중단된 경우, 다른 메모리 사용 작업을
종료한 뒤 다음처럼 재시작한다. `--restart`는 기존 미완료 결과를
`datasets/app_quality_960_q08_v1.interrupted_<시각>_<ID>/`로 보관하고 **처음부터** 다시
처리한다. 완료한 데이터셋은 재시작 대상으로 허용하지 않는다. 보관본은 복구 참고용이며
학습 입력으로 사용하지 않는다. 이미 변환한 이미지도 다시 처리하므로 처리 시간이 필요하다.

```bash
python -m parts.traffic_light.tools.prepare_app_quality --restart --browser-restart-every 25
```

메모리가 작은 환경에서는 `--browser-restart-every 10`으로 더 자주 재시작할 수 있다.
디스크의 학습 데이터 삭제는 RAM 부족의 직접적인 해결책이 아니다. 원본 이미지·기존
manifest·라벨은 전처리에서 재사용하므로 보존한다.

Python JPEG quality=80 대체 경로는 없다. canvas 압축 조건은 실제 앱
`/home/user/gildongmu-test-app/backend/static/js/camera.js`의 `capture()`와
`app.js`의 960/0.8 설정을 기준으로 구현했다. 원본 파일은 사진이므로 브라우저 이미지
디코딩을 사용하고, 실제 앱은 카메라의 video 프레임을 그린다는 차이가 있다.
휴대폰 카메라 ISP, 기기/OS의 색 처리, 모바일 브라우저 빌드까지 동일하게 재현하는 것은
아니다. 브라우저 버전, Playwright 버전, 스크립트 해시를 결과에 남긴다.
[canvas toBlob 문서](https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement/toBlob),
[Playwright 브라우저 설치 문서](https://playwright.dev/python/docs/browsers).

YOLO 라벨은 정규화 xywh이므로 비율 축소 후에도 그대로 유지한다. 픽셀 좌표는
`sx = 새 너비 / 원래 너비`, `sy = 새 높이 / 원래 높이`로 각각 변환하여 반올림된 canvas
크기에 맞춘다. 분류기 manifest의 xyxy에는 이미 10% padding이 있으므로 다시 추가하지 않는다.
작아진 crop을 버리면 학습 표본이 바뀌므로 6px 미만도 유지하고 개수를 기록한다.
평가 시에는 기존 런타임의 6px 미만 색상 분류 생략 규칙을 유지한다.
EXIF 회전이 있는 원본, split 충돌, split 사이의 완전히 동일한 파일 내용은 중단 사유다.
이 경우 원본/라벨의 좌표계나 기존 분할 문제부터 확인한다.

결과 폴더:

```text
datasets/app_quality_960_q08_v1/
├── full_frames/                 # 실제 canvas JPEG, 이미지당 한 번 생성
├── detector_app/                # full_frames 링크 + 동일 정규화 라벨 + data.yaml
├── classifier_app/              # canvas JPEG 디코딩 후 crop, PNG
├── sources.jsonl                # 원본/전송 SHA256, 크기, scale, split
└── summary.json                 # 완료 표시, 브라우저와 설정, 기존 manifest 해시
```

검출기 링크는 같은 결과 폴더의 `full_frames/`를 가리킨다. 라벨과 캐시 경로를 별도 복사본으로
분리하므로 새 학습이 기존 데이터셋에 캐시를 기록하지 않는다. 출력 폴더가 이미 있으면
빈 폴더라도 중단한다. 중단된 실행은 완료된 것으로 취급하지 않으며 새 경로를 쓰거나
위의 `--restart`로 미완료 폴더를 보관한 뒤 재시작한다.

이전 코드로 전처리를 이미 완료했다면 같은 설정의 `detector_app/`, `classifier_app/`,
`summary.json`을 그대로 사용한다. 원본 화질용 폴더가 남아 있어도 학습·평가에서 사용하지
않으며 삭제할 필요가 없다. 원본 화질을 제외하기 위해 전처리를 다시 실행할 필요는 없다.

## 3. 새 모델 학습 (사용자가 실행)

GPU 한 개에서는 아래 두 명령을 순서대로 하나씩 실행한다. `--arm`의 기본값은 `app`이며,
기존 안내의 `--arm app`을 붙여도 된다. `--arm original`은 허용하지 않는다.

```bash
python -m parts.traffic_light.tools.train_app_quality --task detector
python -m parts.traffic_light.tools.train_app_quality --task classifier
```

OOM 때문에 batch/imgsz/AMP를 바꿔야 한다면 과거 실험 설정과 다르다는 점을 기록하고
설정 사본·해시와 출력 경로를 함께 갱신한다.
이미 완료되거나 중단된 run의 audit/output 경로를 덮어쓰거나 resume하지 않는다.

```text
runs/app_quality_960_q08_v1/
├── detector/app/weights/{best,last}.pt
├── classifier/app/mobilenet_v3_small/{best.pt,history.json,result.json}
└── audit/{detector,classifier}_app/
    ├── provenance.json          # 시작 가중치와 해시, 환경, 설정
    ├── args.json
    └── complete.json            # 정상 완료된 best 경로와 해시
```

## 4. 실제 앱 평가셋 라벨 준비

공개 데이터의 test split은 최종 앱 평가셋을 대체하지 않는다. 학습, 모델 선택, 임계값 튜닝에
한 번도 쓰지 않은 촬영 세션 전체를 별도 holdout으로 정한다. 기존에 실패 분석/임계값 선택에
사용한 세션은 개발용으로 남기고 가능하면 새로 촬영한다. 같은 영상/장소/연속 촬영의 인접
프레임이 train/val과 test에 나뉘지 않도록 한다.

앱 서버가 수신한 JPEG 바이트를 그대로 저장해 사용한다. 스크린샷, 검출 박스가 그려진
이미지, 원본 동영상에서 다시 인코딩한 프레임은 사용하지 않는다. 이미 960/0.8 처리가 끝난
앱 JPEG이므로 평가 코드는 다시 축소하거나 JPEG 인코딩하지 않는다.

이 단계는 학습 데이터의 재라벨링이 아니라 실제 앱 평가셋의 정답 준비다. 평가 정답도
이미 있으면 재사용한다. 정답이 없는 전송 JPEG에만 LabelMe rectangle으로
**보이는 모든** 신호등과 횡단보도를 라벨링한다.
신호등은 `R_Signal`, `G_Signal`, `Unknown_Signal`; 횡단보도는 `Zebra_Cross` 또는
`crosswalk`를 사용한다. 음성 프레임도 `shapes: []`인 JSON을 붙인다. 정답 라벨이 없는
프레임을 음성으로 간주하지 않는다. 판독 불가 색상은 unknown으로 표시한다.

```text
/path/to/labeled_app_uploads/
├── unused_session_01/frame_0001.jpg
├── unused_session_01/frame_0001.json
├── unused_session_02/frame_0001.jpg
└── unused_session_02/frame_0001.json
```

```bash
python -m parts.traffic_light.tools.prepare_app_holdout \
  --source /path/to/labeled_app_uploads \
  --sessions unused_session_01 unused_session_02 \
  --output datasets/app_holdout/manifest.jsonl
```

`--sessions`에는 실제로 학습·튜닝에 사용하지 않은 세션만 지정한다. 자동으로 그 사실을
증명할 수는 없다. 변환기는 이미지나 정답을 수정하지 않고 경로와 좌표만 JSONL로 기록한다.
수동 생성 시 한 행은 다음 구조다. xyxy는 **전송 JPEG의 픽셀 좌표**다.

```json
{"image":"/absolute/path/frame.jpg","width":960,"height":540,"session_id":"unused_session_01","origin":"app_upload","split":"test","unused_for_training":true,"annotations":[{"class_name":"pedestrian_signal","xyxy":[400,100,410,120],"color":"red"},{"class_name":"crosswalk","xyxy":[100,300,850,530]}]}
```

평가기는 원본 학습/val/test 이미지 및 새 전송 이미지와 SHA256을 비교하고 완전 중복을
거부한다. 재인코딩된 동일 장면이나 인접 프레임은 해시로 검출할 수 없으므로 촬영 세션의
분리 여부는 사람이 확인해야 한다. 이미지마다 정답 박스가 빠짐없이 있어야 AP/FP가 유효하다.

## 5. 동일 평가셋 비교 (사용자가 실행)

```bash
python -m parts.traffic_light.tools.evaluate_app_quality
```

기존 모델과 새 앱 화질 모델 두 조합을 같은 전송 JPEG, confidence=0.40, NMS IoU=0.70, imgsz=960,
FP32, max_det=300, batch=1로 비교한다. 분류기 confidence=0.60, padding=0.10,
min crop=6, classifier imgsz=224도 같다. 평가셋을 보고 임계값을 따로 조정하지 않는다.

`runs/app_quality_960_q08_v1/evaluation/`에 다음 파일이 생긴다.

- `report.json`: 두 모델 조합의 지표와 `app_minus_historical` 차이
- `frames.jsonl`: 프레임별 GT/예측/색상/GT crop 분류/시간으로 오류 추적 가능
- `provenance.json`: 평가 manifest 해시, 모델 해시, 장치, 패키지, 측정 범위

| 지표 | 정의 |
|---|---|
| 작은 신호등 recall | 전송 JPEG에서 정답 bbox 면적 < 32²px; IoU≥0.5 일대일 매칭 TP / 해당 GT 전체 |
| 신호등·횡단보도 검출 | 운영 confidence에서 TP/FP/FN, precision, recall |
| AP50 / AP50-95 | 별도 confidence=0.001 검출로 IoU 0.50~0.95, 101점 보간; crowd/ignore 없는 전체 라벨 bbox 기준 |
| red→green / green→red | 정답 색별 잘못 반대 색을 낸 개수, 전체 GT 대비 비율, 실제 분류된 GT 대비 비율 |
| 색상 confusion | red/green/unknown 정답 × red/green/unknown/missed/crop_skipped 예측 |
| raw_color | confidence 임계값으로 unknown 처리하기 전 분류 결과도 별도 집계 |
| oracle_color | GT 신호등 박스를 crop한 분류 결과. 검출 누락/위치 오차와 분류기 문제 구분 |
| 추론 시간 | warmup 10회 제외, 프레임당 3회, CUDA 동기화, detector/color/전체 mean·p50·p95 ms |

분모가 0인 지표는 0점으로 꾸미지 않고 null이다. 작은 신호등·각 색·횡단보도 GT 개수와
함께 해석한다. 색상 오분류가 줄었어도 미검출이 늘었다면 개선으로 결론 내리지 않는다.
단일 seed 결과이며 반복 학습 신뢰구간은 제공하지 않는다.

시간은 JPEG 디코딩을 끝낸 프레임에서 검출→검출된 모든 신호등 crop→배치 색상 분류까지다.
모델 로딩·파일 IO·AP용 추가 패스·GT crop 분류·네트워크·브라우저 압축·UI·횡단보도 후보 연결은
포함하지 않는다. 실제 앱은 선택된 신호등 하나만 분류할 수 있으므로 앱 전체 지연과 구별한다.
모델 실행 순서는 프레임마다 순환하며 같은 GPU에 두 모델 조합을 함께 올린다. 다른 GPU
작업 없이 실행한다. `report.json`의 시간 차이 단위는 ms, recall/AP/오분류율 차이는 0~1
비율(×100하면 %p)이다.

## 구현 검증 범위

합성 데이터로 축소 반올림, 미확대, 전체 이미지 처리 후 crop, 원본 보존, 라벨/split/음성
샘플 유지, 출력 재사용 거부, GT 매칭, AP, 색상 분모, 중복 평가 이미지 거부를 테스트한다.
실제 학습·평가 결과는 [전체 분할 평가](DATASET_SPLIT_EVALUATION_20260919.md)에 기록했다.
이 단위 테스트 자체는 실제 모델을 학습하지 않는다. 브라우저 smoke test에는
Playwright와 Chromium 시스템 라이브러리 설치가 필요하다.
