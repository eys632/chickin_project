# 닭 폐사체 감지 시스템 (Chicken Mortality Detection System)

## 프로젝트 개요

이 프로젝트는 YOLO 객체 탐지 모델과 컴퓨터 비전 기법을 결합하여 닭장 내 닭의 움직임을 분석하고, 움직이지 않는 닭(폐사체 의심)을 자동으로 감지하는 시스템입니다. 복수의 GPU를 활용한 병렬 처리를 지원하여 대용량 비디오를 효율적으로 분석할 수 있습니다.

## 주요 기능

- **실시간 닭 탐지**: YOLO v8 모델을 사용한 고정밀 닭 객체 탐지
- **움직임 분석**: 배경 차분법과 정적 픽셀 분석을 통한 움직임 감지
- **객체 추적**: IoU 기반 다중 객체 추적으로 개체별 상태 모니터링
- **폐사체 감지**: 일정 시간 이상 움직이지 않는 닭을 폐사체로 판정
- **병렬 처리**: 멀티 GPU 지원으로 대용량 비디오 고속 처리
- **시각화**: 실시간 결과 시각화 및 분석 리포트 생성

## 파일 구조

```
├── 0508_main_detection.py      # 메인 감지 프로그램 (단일 GPU)
├── 0508_utils.py              # 비디오 세그먼트 처리 유틸리티
├── 0508_simple_ver.py         # 간소화된 버전 (지역별 분할 처리)
├── multi_gpu_extension.py     # 멀티 GPU 확장 모듈
└── README.md                  # 프로젝트 문서
```

## 각 파일 상세 설명

### 1. `0508_main_detection.py` - 메인 감지 프로그램

**목적**: 완전한 기능을 갖춘 메인 감지 프로그램으로, 단일 GPU에서 전체 비디오를 처리합니다.

**주요 특징**:
- MOG2 배경 서브트랙터를 이용한 전경/배경 분리
- 프레임 간 차이 분석을 통한 정적 픽셀 감지
- YOLO를 이용한 닭 객체 탐지 및 바운딩 박스 추출
- IoU 기반 객체 추적 시스템
- 정적 상태 판정 알고리즘 (IoU > 0.9, 정적 픽셀 비율 > 0.8)
- 실시간 시각화 및 결과 저장

**알고리즘 흐름**:
1. 배경 모델 학습 및 전경 마스크 생성
2. 프레임 간 차이로 정적 픽셀 마스크 생성
3. YOLO로 닭 객체 탐지
4. IoU 매칭으로 객체 ID 할당
5. 바운딩 박스 내 정적 픽셀 비율 계산
6. 10프레임 이상 정적 상태 유지시 폐사체로 판정

### 2. `0508_utils.py` - 비디오 세그먼트 처리 유틸리티

**목적**: 비디오를 특정 프레임 구간으로 나누어 처리하는 모듈화된 함수를 제공합니다.

**주요 기능**:
- 지정된 프레임 범위 처리 (`start_frame` ~ `end_frame`)
- GPU 장치별 독립적인 ID 할당 (멀티 GPU 처리 대비)
- 세그먼트별 결과 저장 및 리포트 생성
- 명령행 인터페이스 지원

**사용 사례**:
```python
process_video_segment(
    video_path="/path/to/video.mp4",
    model_path="/path/to/model.pt", 
    save_dir="/path/to/output",
    start_frame=0,
    end_frame=1000,
    device=0
)
```

### 3. `0508_simple_ver.py` - 간소화된 버전

**목적**: 계산 복잡도를 줄이고 처리 속도를 향상시킨 간소화된 감지 알고리즘입니다.

**주요 특징**:
- 화면을 4개 구역으로 분할하여 병렬 처리
- 시간 간격 기반 샘플링 (60초 간격)
- 중심점 기반 움직임 분석 (반경 3픽셀 허용)
- 배치 프레임 처리 (100프레임 단위)
- 지역별 독립적인 결과 생성

**알고리즘**:
1. 화면을 4개 영역으로 분할 (좌상, 우상, 좌하, 우하)
2. 각 영역에서 60초 간격으로 100프레임 연속 분석
3. 중심점이 반경 3픽셀 내에서 유지되는 객체를 정적으로 판정
4. 지역별 결과 병합 및 시각화

### 4. `multi_gpu_extension.py` - 멀티 GPU 확장 모듈

**목적**: 여러 GPU를 활용하여 대용량 비디오를 병렬 처리하는 분산 컴퓨팅 모듈입니다.

**주요 기능**:
- PyTorch 분산 처리 (`torch.distributed`) 활용
- 비디오를 GPU 수만큼 균등 분할
- GPU별 독립적인 처리 후 결과 병합
- 자동 비디오/로그/리포트 통합

**처리 과정**:
1. 전체 프레임을 GPU 수로 균등 분할
2. 각 GPU에서 할당된 프레임 범위 처리
3. 모든 GPU 처리 완료 후 결과 병합
4. 통합 비디오 및 분석 리포트 생성

## 시스템 요구사항

### 하드웨어
- NVIDIA GPU (CUDA 지원)
- 충분한 GPU 메모리 (최소 8GB 권장)
- 멀티 GPU 처리시 A100 등 고성능 GPU 권장

### 소프트웨어
```
Python >= 3.8
PyTorch >= 1.10.0
OpenCV >= 4.5.0
Ultralytics YOLO >= 8.0.0
NumPy >= 1.21.0
Pandas >= 1.3.0
Matplotlib >= 3.4.0
```

## 설치 및 실행

### 1. 환경 설정
```bash
pip install torch torchvision opencv-python ultralytics numpy pandas matplotlib
```

### 2. 단일 GPU 실행
```bash
python 0508_main_detection.py
```

### 3. 간소화 버전 실행
```bash
python 0508_simple_ver.py
```

### 4. 멀티 GPU 실행
```bash
python multi_gpu_extension.py --video /path/to/video.mp4 --model /path/to/model.pt --output /path/to/output
```

### 5. 세그먼트 처리
```bash
python 0508_utils.py --video /path/to/video.mp4 --model /path/to/model.pt --output /path/to/output --start 0 --end 1000 --device 0
```

## 주요 파라미터

### 감지 파라미터
- `static_threshold`: 정적 픽셀 판정 임계값 (기본값: 8)
- `iou_threshold`: IoU 정적 판정 임계값 (기본값: 0.9)
- `mask_ratio_threshold`: 정적 픽셀 비율 임계값 (기본값: 0.8)
- `yolo_conf`: YOLO 신뢰도 임계값 (기본값: 0.3)

### 처리 파라미터
- `learning_rate`: 배경 모델 학습률 (기본값: 0.01)
- `min_static_area`: 최소 정적 영역 크기 (기본값: 200픽셀)
- `TOLERANCE_RADIUS`: 중심점 허용 반경 (간소화 버전, 기본값: 3픽셀)

## 출력 결과

### 생성 파일
- `output.mp4`: 분석 결과가 오버레이된 비디오
- `detection_log.csv`: 프레임별 상세 감지 로그
- `report.txt`: 분석 요약 리포트
- `static_status_graph.png`: 시간별 정적 상태 변화 그래프
- `frame_XXXXXX.jpg`: 30프레임 간격 스냅샷 이미지

### CSV 로그 형식
```csv
frame,chicken_id,x1,y1,x2,y2,static_ratio,iou,is_static
1,1001,100,150,200,250,0.75,0.95,0
2,1001,102,152,198,248,0.85,0.92,1
```

## 알고리즘 상세

### 1. 정적 픽셀 감지
```python
# 프레임 간 차이 계산
frame_diff = cv2.absdiff(current_gray, prev_gray)

# 임계값 적용 (차이가 작은 픽셀 = 정적)
static_mask = cv2.threshold(frame_diff, static_threshold, 255, cv2.THRESH_BINARY_INV)[1]
```

### 2. 객체 추적
```python
# IoU 기반 ID 매칭
def calculate_iou(box1, box2):
    intersection = (x_right - x_left) * (y_bottom - y_top)
    union = box1_area + box2_area - intersection
    return intersection / union
```

### 3. 폐사체 판정
```python
# 정적 상태 판정 조건
if iou > iou_threshold and static_ratio > mask_ratio_threshold:
    static_counter[chicken_id] += 1
    if static_counter[chicken_id] >= 10:  # 10프레임 이상
        is_static = True
```

## 성능 최적화

### 단일 GPU 최적화
- 배경 서브트랙터 히스토리 조정
- YOLO 추론 배치 크기 최적화
- 불필요한 메모리 복사 최소화

### 멀티 GPU 최적화
- 비디오 세그먼트 균등 분할
- GPU 간 통신 최소화
- 메모리 효율적인 결과 병합

## 확장 가능성

1. **실시간 처리**: 웹캠/RTSP 스트림 지원
2. **클라우드 배포**: Docker 컨테이너화
3. **모바일 최적화**: TensorRT/ONNX 변환
4. **알림 시스템**: 폐사체 감지시 자동 알림
5. **웹 인터페이스**: Flask/FastAPI 기반 웹 UI

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 개발되었습니다.

## 기여자

- 개발자: a202192020
- 지도교수: 송교수님
- 모델 제공: 양륜비박사님

## 문의사항

프로젝트 관련 문의사항이나 개선 제안은 이슈로 등록해 주세요.
