# 🐔 Chickin Dead Chicken Detection - 0513 버전

닭 사육장에서 촬영한 영상을 분석하여 **움직이지 않는 닭(사망 추정)**을 자동으로 탐지하고, 그 좌표와 ID를 기록하는 프로젝트입니다.  
영상은 3x3 그리드로 분할되며, 각각의 영역에 YOLO 모델을 적용하여 객체를 감지하고, 일정 시간 동안 움직이지 않으면 죽은 닭으로 판단하여 CSV에 기록합니다.
## 시스템 구조
![image](https://github.com/user-attachments/assets/98b957f3-0c24-4f64-9cf3-eba608d1eca7)
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

## 수정해야 할 문제점
 - 시간이 지날수록 닭 탐지 오류가 증가함
 - 닭이 아닌 픽셀들을 닭이라고 인식하는 박스가 점점 많아짐
   
![닭 탐지 시연](https://user-images.githubusercontent.com/https://private-user-images.githubusercontent.com/87321507/445644526-9ce1d02a-731f-4bf1-b5d2-ea1d1fa53284.gif?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDc3NTAyNTIsIm5iZiI6MTc0Nzc0OTk1MiwicGF0aCI6Ii84NzMyMTUwNy80NDU2NDQ1MjYtOWNlMWQwMmEtNzMxZi00YmYxLWI1ZDItZWExZDFmYTUzMjg0LmdpZj9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTIwVDE0MDU1MlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWFkZWRiZTA2NzkxYmZjMDVjZjdjMWFmMzdjYzYwMDYyMzI4N2JkY2EwZTRjMDUxZDVhODBiY2U4YmE4ZThhMjEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.sdnldWbfgu8PG379nEMrKUX4T2fNlHxWZ2Oeih7Vcn4)













