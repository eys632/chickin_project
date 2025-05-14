# 🐔 Chickin Dead Chicken Detection - 0513 버전

닭 사육장에서 촬영한 영상을 분석하여 **움직이지 않는 닭(사망 추정)**을 자동으로 탐지하고, 그 좌표와 ID를 기록하는 프로젝트입니다.  
영상은 3x3 그리드로 분할되며, 각각의 영역에 YOLO 모델을 적용하여 객체를 감지하고, 일정 시간 동안 움직이지 않으면 죽은 닭으로 판단하여 CSV에 기록합니다.

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















