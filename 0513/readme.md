# 🐔 Chickin Dead Chicken Detection - 0513 버전

닭 사육장에서 촬영한 영상을 분석하여 **움직이지 않는 닭(사망 추정)**을 자동으로 탐지하고, 그 좌표와 ID를 기록하는 프로젝트입니다.  
영상은 3x3 그리드로 분할되며, 각각의 영역에 YOLO 모델을 적용하여 객체를 감지하고, 일정 시간 동안 움직이지 않으면 죽은 닭으로 판단하여 CSV에 기록합니다.
## 시스템 구조
```
<svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">
  <!-- 배경 -->
  <rect width="800" height="600" fill="#f8f9fa" />
  
  <!-- 제목 -->
  <text x="400" y="40" font-family="Arial" font-size="24" text-anchor="middle" font-weight="bold">움직이지 않는 닭 감지 및 추적 시스템 구조도</text>
  
  <!-- 입력 단계 -->
  <rect x="50" y="80" width="150" height="60" rx="10" fill="#ffcccb" stroke="#ff6b6b" stroke-width="2" />
  <text x="125" y="115" font-family="Arial" font-size="14" text-anchor="middle" font-weight="bold">입력 영상</text>
  <text x="125" y="135" font-family="Arial" font-size="12" text-anchor="middle">(777.mp4)</text>
  
  <!-- 1단계: 영상 분할 -->
  <rect x="300" y="80" width="200" height="70" rx="10" fill="#c2e0ff" stroke="#0066cc" stroke-width="2" />
  <text x="400" y="110" font-family="Arial" font-size="14" text-anchor="middle" font-weight="bold">1단계: 영상 분할 (10%)</text>
  <text x="400" y="130" font-family="Arial" font-size="12" text-anchor="middle">video_splitter.py</text>
  
  <!-- 분할 영역 결과 -->
  <g transform="translate(600, 65)">
    <rect x="0" y="0" width="30" height="30" fill="#e6f2ff" stroke="#0066cc" stroke-width="1" />
    <rect x="35" y="0" width="30" height="30" fill="#e6f2ff" stroke="#0066cc" stroke-width="1" />
    <rect x="70" y="0" width="30" height="30" fill="#e6f2ff" stroke="#0066cc" stroke-width="1" />
    <rect x="0" y="35" width="30" height="30" fill="#e6f2ff" stroke="#0066cc" stroke-width="1" />
    <rect x="35" y="35" width="30" height="30" fill="#e6f2ff" stroke="#0066cc" stroke-width="1" />
    <rect x="70" y="35" width="30" height="30" fill="#e6f2ff" stroke="#0066cc" stroke-width="1" />
    <rect x="0" y="70" width="30" height="30" fill="#e6f2ff" stroke="#0066cc" stroke-width="1" />
    <rect x="35" y="70" width="30" height="30" fill="#e6f2ff" stroke="#0066cc" stroke-width="1" />
    <rect x="70" y="70" width="30" height="30" fill="#e6f2ff" stroke="#0066cc" stroke-width="1" />
    <text x="50" y="120" font-family="Arial" font-size="12" text-anchor="middle">9개 분할 영역</text>
  </g>
  
  <!-- 화살표: 입력 -> 분할 -->
  <path d="M200 110 L300 110" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrow)" />
  
  <!-- 화살표: 분할 -> 분할영역 -->
  <path d="M500 110 L590 110" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrow)" />
  
  <!-- 2단계: YOLO 추론 -->
  <rect x="300" y="200" width="200" height="70" rx="10" fill="#d7f9d7" stroke="#28a745" stroke-width="2" />
  <text x="400" y="230" font-family="Arial" font-size="14" text-anchor="middle" font-weight="bold">2단계: YOLO 추론 (40%)</text>
  <text x="400" y="250" font-family="Arial" font-size="12" text-anchor="middle">yolo_model_loader.py</text>
  
  <!-- YOLO 모델 -->
  <rect x="50" y="205" width="150" height="60" rx="10" fill="#ffe6cc" stroke="#ff9900" stroke-width="2" />
  <text x="125" y="240" font-family="Arial" font-size="14" text-anchor="middle" font-weight="bold">YOLO 모델</text>
  <text x="125" y="255" font-family="Arial" font-size="12" text-anchor="middle">(best.pt)</text>
  
  <!-- 화살표: 모델 -> YOLO 추론 -->
  <path d="M200 235 L300 235" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrow)" />
  
  <!-- 화살표: 분할영역 -> YOLO 추론 -->
  <path d="M650 150 L650 175 L400 175 L400 200" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrow)" />
  
  <!-- 3단계: 움직임 추적 -->
  <rect x="300" y="320" width="200" height="70" rx="10" fill="#ffefc2" stroke="#ffc107" stroke-width="2" />
  <text x="400" y="350" font-family="Arial" font-size="14" text-anchor="middle" font-weight="bold">3단계: 움직임 추적 (50%)</text>
  <text x="400" y="370" font-family="Arial" font-size="12" text-anchor="middle">motion_tracker.py</text>
  
  <!-- 화살표: YOLO 추론 -> 움직임 추적 -->
  <path d="M400 270 L400 320" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrow)" />
  
  <!-- CSV 로깅 모듈 -->
  <rect x="50" y="390" width="150" height="60" rx="10" fill="#e6e6ff" stroke="#6610f2" stroke-width="2" />
  <text x="125" y="420" font-family="Arial" font-size="14" text-anchor="middle" font-weight="bold">CSV 로깅</text>
  <text x="125" y="435" font-family="Arial" font-size="12" text-anchor="middle">csv_logger.py</text>
  
  <!-- 화살표: 움직임 추적 -> CSV 로깅 -->
  <path d="M300 370 L200 370 L200 410 L200 410" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrow)" />
  
  <!-- 4단계: 비디오 병합 -->
  <rect x="300" y="440" width="200" height="70" rx="10" fill="#f8d7da" stroke="#dc3545" stroke-width="2" />
  <text x="400" y="470" font-family="Arial" font-size="14" text-anchor="middle" font-weight="bold">4단계: 비디오 병합 및 시각화</text>
  <text x="400" y="490" font-family="Arial" font-size="12" text-anchor="middle">video_merger.py</text>
  
  <!-- 화살표: 움직임 추적 -> 비디오 병합 -->
  <path d="M400 390 L400 440" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrow)" />
  
  <!-- 결과 영상 -->
  <rect x="600" y="440" width="150" height="70" rx="10" fill="#d9d9d9" stroke="#343a40" stroke-width="2" />
  <text x="675" y="465" font-family="Arial" font-size="14" text-anchor="middle" font-weight="bold">결과 영상</text>
  <text x="675" y="485" font-family="Arial" font-size="11" text-anchor="middle">merged_dead_chickens.mp4</text>
  <text x="675" y="500" font-family="Arial" font-size="11" text-anchor="middle">part_i_j_processed.mp4</text>
  
  <!-- 화살표: 비디오 병합 -> 결과 영상 -->
  <path d="M500 475 L600 475" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrow)" />
  
  <!-- 결과 CSV 파일 -->
  <rect x="600" y="360" width="150" height="70" rx="10" fill="#d9d9d9" stroke="#343a40" stroke-width="2" />
  <text x="675" y="385" font-family="Arial" font-size="14" text-anchor="middle" font-weight="bold">결과 CSV 파일</text>
  <text x="675" y="405" font-family="Arial" font-size="11" text-anchor="middle">dead_chickens.csv</text>
  <text x="675" y="420" font-family="Arial" font-size="11" text-anchor="middle">id_mapping.csv</text>
  
  <!-- 화살표: CSV 로깅 -> 결과 CSV 파일 -->
  <path d="M200 410 L250 410 L350 410 L600 395" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrow)" />
  
  <!-- main.py 전체 제어 -->
  <rect x="300" y="540" width="200" height="40" rx="10" fill="#17a2b8" stroke="#138496" stroke-width="2" />
  <text x="400" y="565" font-family="Arial" font-size="14" text-anchor="middle" font-weight="bold" fill="white">main.py (전체 제어)</text>
  
  <!-- 화살표 마커 정의 -->
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#666" />
    </marker>
  </defs>
  
  <!-- 연결선: main.py -> 모든 단계 -->
  <path d="M400 540 L400 520 L250 520 L250 300 L270 300" stroke="#17a2b8" stroke-width="2" stroke-dasharray="5,5" fill="none" />
  <path d="M250 300 L250 110 L300 110" stroke="#17a2b8" stroke-width="2" stroke-dasharray="5,5" fill="none" />
  <path d="M250 300 L250 235 L300 235" stroke="#17a2b8" stroke-width="2" stroke-dasharray="5,5" fill="none" />
  <path d="M250 300 L250 355 L300 355" stroke="#17a2b8" stroke-width="2" stroke-dasharray="5,5" fill="none" />
  <path d="M250 300 L250 475 L300 475" stroke="#17a2b8" stroke-width="2" stroke-dasharray="5,5" fill="none" />
</svg>
```
## 📂 프로젝트 구조 (0513 폴더 기준)
```
├── main.py # 전체 실행 파이프라인
├── csv_logger.py # 죽은 닭 정보 CSV로 기록
├── motion_tracker.py # 움직임 추적 및 사망 판단
├── video_splitter.py # 입력 영상을 3x3으로 분할
├── video_merger.py # 분석 결과를 시각화한 영상으로 병합
├── yolo_model_loader.py # YOLO 모델 로드 및 객체 탐지 수행
```
## 🔧 요구 사항

- Python 3.8 이상
- OpenCV
- NumPy
- Ultralytics YOLOv8
- tqdm

필요한 라이브러리를 설치하려면 다음 명령어를 사용하세요:

```
pip install opencv-python numpy ultralytics tqdm
```
## 🚀 실행 방법

1. `main.py` 파일을 열고 아래 두 경로를 본인의 환경에 맞게 수정하세요:

```python
video_path = "/절대경로/777.mp4"        # 분석할 CCTV 영상 경로
model_path = "/절대경로/best.pt"       # 학습된 YOLOv8 모델 경로
```
2. 터미널에서 실행합니다:
```
python main.py
```
3. 실행이 완료되면 /0513/dead_check_날짜시간/ 폴더가 자동 생성되고, 아래와 같은 결과 파일들이 저장됩니다.
## 📝 출력 결과

| 파일명                  | 설명 |
|-------------------------|------|
| `dead_chickens.csv`     | 죽은 닭의 프레임 번호, ID, 중심 좌표(cx, cy), 위치(part 이름) 기록 |
| `id_mapping.csv`        | 내부에서 새로 부여한 ID와 원래 YOLO ID 간의 매핑 정보 |
| `output_video.mp4` 또는 `.avi` | 박스와 ID가 시각화된 전체 병합 영상 |
## 📌 로직 개요

1. **영상 분할**
   - `video_splitter.py`: 입력 CCTV 영상을 3x3 그리드로 분할합니다.
   - 결과: part_0_0.mp4 ~ part_2_2.mp4 총 9개 영상 생성

2. **객체 탐지**
   - `yolo_model_loader.py`: 각 분할 영상에 YOLOv8 모델을 적용해 닭을 감지합니다.
   - 감지된 닭의 중심 좌표를 추출합니다.

3. **좌표 추적 및 사망 판단**
   - `motion_tracker.py`: 닭의 위치를 ID별로 추적하고,
     지정된 프레임 수(기본 1800프레임 = 약 1분) 이상 움직임이 없으면 사망으로 간주합니다.

4. **결과 기록 및 시각화**
   - `csv_logger.py`: 사망한 닭 정보를 `dead_chickens.csv`에 저장합니다.
   - `video_merger.py`: 분할 영상들을 병합하여 박스, ID, 위치가 표시된 결과 영상(`output_video.mp4`)을 생성합니다.
## 🧠 핵심 아이디어

- **3x3 영상 분할**을 통해 해상도 저하 없이 탐지 정확도를 향상시킵니다.
- YOLOv8을 활용한 **닭 객체 탐지**와 중심 좌표 기반 **움직임 추적**을 결합합니다.
- 특정 시간 동안 **위치 변화가 없는 객체를 죽은 닭으로 추정**합니다.
- 객체의 위치와 ID를 시각적으로 병합 영상에 표시하여 **사육장 내 상태를 직관적으로 파악**할 수 있습니다.
## 📌 주의 사항

- 입력 영상은 **정면 고정 카메라**로 촬영된 사육장 CCTV여야 정확한 판단이 가능합니다.
- 영상의 **프레임 수가 너무 적거나 FPS가 낮을 경우**, 판단 정확도가 떨어질 수 있습니다.
- YOLO 모델은 반드시 **닭을 정확히 감지할 수 있도록 사전 학습된 YOLOv8 모델**이어야 합니다.
- 영상 해상도가 너무 낮거나 분할된 화면에 닭이 너무 작게 나타나는 경우, 탐지가 어려울 수 있습니다.

---

## 💬 문의 또는 피드백

- 이 시스템은 테스트 목적이며, 실제 사육장 환경에 적용하려면 **정밀 조정 및 현장 실험이 필요**합니다.
- 사용 중 문제나 개선 아이디어가 있다면 공유해주세요!















