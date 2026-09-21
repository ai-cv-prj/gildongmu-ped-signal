# 기존 분할 전체 평가 — 2026-09-19

기존 검출기와 앱 화질 재학습 검출기에 동일한 train·val·test 이미지를 입력해 실제 추론한 결과다. 모델 학습·데이터 재전처리는 실행하지 않았다.

## 데이터와 기준

- 입력: 기존 분할을 보존한 Chromium canvas 긴 변 최대 960px, JPEG quality=0.8 이미지.
- train 36,204장, val 559장, test 45장 전체를 사용했다. 표본 추출이나 분할 변경은 없다.
- train은 학습 적합도, val은 학습 중 체크포인트 선택에 사용한 검증 성능, test는 학습/선택에 사용하지 않은 분할의 성능이다.
- 원본 이미지 SHA-256 기준 분할 간 완전 중복은 0건이다. 유사 장면이나 연속 촬영의 중복까지 판정한 것은 아니다.
- 원본 manifest와 모든 평가 라벨의 해시를 검사했다. val/test 604장의 평가 GT 좌표를 YOLO 라벨과 대조했고 최대 오차는 0.000137px였다.
- 전체 AP: Ultralytics 8.4.150 공식 validator, confidence=0.001, imgsz=960, batch=16, rect=True, FP32, NMS IoU=0.7, max_det=300, augmentation 없음.
- 고정 임계값 지표: confidence=0.4, GT matching IoU=0.5, 클래스별 confidence 내림차순 일대일 매칭.
- 작은 신호등: 모델 letterbox 전 압축 이미지에서 GT 면적 <1,024px². 가로·세로 각각 32px 미만이라는 의미는 아니다.
- 색상 분류기는 이번 평가에 포함하지 않았다. 아래는 신호등·횡단보도 검출 성능이다.

## 전체 mAP 비교

| 분할 | 이미지 | 기존 mAP50 | 새 mAP50 | 기존 mAP50–95 | 새 mAP50–95 | 변화 |
|---|---:|---:|---:|---:|---:|---:|
| train | 36,204 | 97.86% | 97.99% | 85.07% | 86.20% | +1.13%p |
| val | 559 | 86.16% | 90.12% | 65.56% | 69.91% | +4.36%p |
| test | 45 | 76.23% | 80.54% | 52.51% | 55.53% | +3.02%p |

mAP는 신호등·횡단보도 두 클래스의 평균이다. mAP50–95는 IoU 0.50부터 0.95까지 평균하므로 박스 위치까지 더 엄격하게 평가한다.

## 클래스별 AP

| 분할 | 모델 | 클래스 | GT 수 | AP50 | AP50–95 |
|---|---|---|---:|---:|---:|
| train | historical | pedestrian_signal | 13594 | 96.91% | 76.63% |
| train | historical | crosswalk | 35929 | 98.81% | 93.51% |
| train | app | pedestrian_signal | 13594 | 96.97% | 77.04% |
| train | app | crosswalk | 35929 | 99.00% | 95.36% |
| val | historical | pedestrian_signal | 808 | 87.67% | 62.07% |
| val | historical | crosswalk | 1059 | 84.64% | 69.04% |
| val | app | pedestrian_signal | 808 | 91.39% | 63.66% |
| val | app | crosswalk | 1059 | 88.84% | 76.16% |
| test | historical | pedestrian_signal | 120 | 83.38% | 54.59% |
| test | historical | crosswalk | 121 | 69.07% | 50.44% |
| test | app | pedestrian_signal | 120 | 88.59% | 54.08% |
| test | app | crosswalk | 121 | 72.49% | 56.98% |

## confidence 0.4에서의 검출 성능

| 분할 | 모델 | 클래스 | TP / FP / FN | Precision | Recall |
|---|---|---|---:|---:|---:|
| train | historical | pedestrian_signal | 12928 / 1209 / 666 | 91.45% | 95.10% |
| train | historical | crosswalk | 35365 / 715 / 564 | 98.02% | 98.43% |
| train | app | pedestrian_signal | 13212 / 1559 / 382 | 89.45% | 97.19% |
| train | app | crosswalk | 35489 / 704 / 440 | 98.05% | 98.78% |
| val | historical | pedestrian_signal | 448 / 16 / 360 | 96.55% | 55.45% |
| val | historical | crosswalk | 582 / 12 / 477 | 97.98% | 54.96% |
| val | app | pedestrian_signal | 500 / 24 / 308 | 95.42% | 61.88% |
| val | app | crosswalk | 609 / 9 / 450 | 98.54% | 57.51% |
| test | historical | pedestrian_signal | 36 / 0 / 84 | 100.00% | 30.00% |
| test | historical | crosswalk | 46 / 1 / 75 | 97.87% | 38.02% |
| test | app | pedestrian_signal | 47 / 0 / 73 | 100.00% | 39.17% |
| test | app | crosswalk | 55 / 0 / 66 | 100.00% | 45.45% |

Precision은 검출 결과 중 정답 비율, recall은 전체 정답 중 찾아낸 비율이다. JSON의 `official.Box-P/Box-R`는 검증기가 F1 기준으로 선택한 임계값의 값이므로 이 표의 고정 confidence 0.4 값과 섞어서 해석하지 않는다.

## 작은 신호등 recall

| 분할 | 작은 신호등 GT | 기존 검출 / recall | 새 검출 / recall | 변화 |
|---|---:|---:|---:|---:|
| train | 10985 | 10407 / 94.74% | 10655 / 97.00% | +2.26%p |
| val | 656 | 310 / 47.26% | 355 / 54.12% | +6.86%p |
| test | 96 | 17 / 17.71% | 27 / 28.12% | +10.42%p |

## 검출기 시간

GPU는 NVIDIA GeForce RTX 5080 Laptop GPU다. 아래는 batch=16 검증에서 이미지당 평균 순수 모델 추론 시간이다. 단일 이미지 앱 응답시간이 아니며, 파일 읽기·JPEG 디코딩·전처리·후처리·색상 분류·네트워크를 제외한다. 각 모델/분할 1회 전체 평가의 측정값이며 작은 차이에 의미를 부여하지 않는다.

| 분할 | 기존 | 새 모델 |
|---|---:|---:|
| train | 2.66ms | 2.66ms |
| val | 2.70ms | 2.75ms |
| test | 3.41ms | 3.32ms |

## 해석과 한계

- val/test에서 새 모델의 성능이 개선됐다. 하지만 test의 신호등 recall 39.17%, 작은 신호등 recall 28.12%로 누락 문제는 남아 있다.
- 모든 지표가 개선된 것은 아니다. test 신호등 AP50–95는 54.59% → 54.08%로 소폭 하락했다. 신호등 검출 recall 개선이 엄격한 박스 위치 정확도의 개선까지 의미하지는 않는다. train/val 신호등 precision도 각각 91.45% → 89.45%, 96.55% → 95.42%로 낮아졌다.
- test는 45장이고 train/val과 원본 폴더가 구분된다(train: bbox_1·2, val: bbox_3·4, test: bbox_5·6). 분할 간 난이도/분포 차이와 작은 표본을 함께 고려해야 한다. train–test 점수 차이만으로 과적합의 원인이나 정도를 확정하지 않는다.
- 이전 1장씩 앱 파이프라인 평가와는 배치 letterbox 형상 및 AP 집계 구현이 달라 수치가 다르다. 이번 표 안에서는 두 모델 모두 같은 검증 방식을 적용했다.
- 기존 검출기는 단계적 추가학습 이력이고 새 검출기는 원본 사전학습 가중치에서 시작했다. 따라서 개선 전부를 압축 화질 한 변수의 인과효과로 단정하지 않는다.
- 실제 앱 촬영 이미지의 성능이나 빨강/초록 분류 정확도를 이 평가에서 결론 내리지 않는다.

## 재현 명령과 저장 경로

```bash
.venv/bin/python -m parts.traffic_light.tools.evaluate_dataset_splits \
  --output runs/app_quality_960_q08_v1/evaluation_splits_new \
  --splits val test train --models historical app --batch 16 --workers 4
```

- 기존 `.venv`의 torch, torchvision, ultralytics, OpenCV, Pillow, PyYAML 환경을 사용한다. 버전은 결과 JSON에 기록했다.
- 전체 결과·설정·가중치 해시: `runs/app_quality_960_q08_v1/evaluation_splits_20260919_final/report.json`
- 모델/분할별 결과: 같은 폴더의 `{train,val,test}_{historical,app}.json`
- 이미지별 정답과 confidence≥0.4 예측: 같은 폴더의 `{split}_{model}_frames.jsonl`
- 여섯 평가의 이미지별 결과 재집계 확인: 같은 폴더의 `integrity_audit.json`
- 평가 코드: `parts/traffic_light/tools/evaluate_dataset_splits.py`
- 최초 시도의 JSON 숫자 형식 오류로 남은 `evaluation_splits_20260919/`는 미완료 결과여서 2026-09-19에 삭제했다. 비교에는 보존된 `_final/`만 사용한다.
- Git에 보존한 지표 사본: [dataset_split_metrics.json](reports/20260919/dataset_split_metrics.json)
