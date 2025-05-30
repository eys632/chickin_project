# 🐔 닭 움직임 및 정적 상태 분석 (특정 영역 테스트 버전)

이 프로젝트는 동영상 내에서 특정 관심 영역(Region of Interest, ROI)에 있는 **닭의 움직임을 분석하고, 닭이 장시간 정지해 있을 경우 '정적' 상태를 넘어 '죽은 닭'으로 판정**하는 기능을 테스트하는 Python 스크립트입니다. `YOLOv8` 모델을 활용한 닭 탐지, 배경 서브트랙션(Background Subtraction)을 통한 움직임 감지, 그리고 트래킹(Tracking) 기술을 결합하여 닭의 상태를 실시간으로 시각화합니다.

이 `README`는 코드의 핵심 기능, 동작 방식, 그리고 실행 방법을 명확하고 쉽게 설명합니다.

---

## 🚀 주요 기능 및 개념

이 프로젝트는 다음과 같은 핵심 기능과 개념을 사용합니다:

1.  **동영상 처리**: `OpenCV` 라이브러리를 사용하여 동영상을 프레임 단위로 읽고 처리하며, 분석 결과를 다시 동영상 파일로 저장해요.
2.  **YOLOv8 객체 탐지**: `ultralytics` 라이브러리의 `YOLOv8` 모델을 사용하여 동영상 프레임 내에서 닭의 위치를 정확하게 탐지합니다.
3.  **배경 서브트랙션 (Background Subtraction)**: `cv2.createBackgroundSubtractorMOG2`를 사용하여 움직이는 객체(닭)와 정적인 배경을 분리해요. 이를 통해 닭의 움직임 여부를 판단하는 데 필요한 **`정적 마스크`**를 생성합니다.
4.  **닭 트래킹**: 탐지된 닭들이 프레임이 바뀌어도 동일한 개체로 인식될 수 있도록 **`IoU (Intersection over Union)` 기반의 간단한 트래킹 로직**을 적용합니다. 각 닭에게는 고유한 ID가 부여돼요.
5.  **'정적' 상태 판정**:
    * 닭의 바운딩 박스(Bounding Box) 내에서 배경 서브트랙션으로 얻은 **`정적 픽셀 비율`**을 계산합니다.
    * 이 비율이 특정 임계값(`STATIC_RATIO_THRESHOLD`)을 초과하고, `CONSECUTIVE_STATIC_FRAMES`만큼 연속된 프레임 동안 정지해 있으면 해당 닭을 '정적' 상태로 판정해요.
6.  **'죽은 닭' 판정**:
    * '정적' 상태가 `DEAD_CHICKEN_MIN_DURATION_MINUTES`로 설정된 최소 시간(분) 이상 지속되면, 해당 닭을 **'죽은 닭'**으로 최종 판정합니다.
7.  **다중 관심 영역 (ROI) 분석**:
    * 원본 동영상을 3x3 그리드로 가상 분할한 뒤, 사용자가 지정한 **두 개의 특정 영역(`왼쪽 가운데`와 `오른쪽 가장 위`)에만** 위 분석 로직을 적용합니다.
    * 각 영역은 독립적으로 닭을 탐지하고 트래킹하며 상태를 판정해요.
8.  **시각화 오버레이**:
    * 분석된 닭의 바운딩 박스(ID, 상태 표시), 정적 마스크, 그리고 각 ROI의 경계선을 **원본 동영상 프레임 위에 실시간으로 오버레이**하여 새로운 결과 동영상을 생성합니다.
    * 닭의 상태에 따라 바운딩 박스 색상이 달라져요 (움직임: 초록색, 정적: 파란색, 죽음: 빨간색).

---

## 🛠️ 환경 설정 및 실행 방법

이 프로젝트를 실행하기 위해 다음 단계를 따라주세요.

### 1. 필수 라이브러리 설치

Python 환경에서 다음 라이브러리들을 설치해야 합니다:

```
pip install opencv-python numpy ultralytics tqdm
```
### 2. 코드 다운로드
이 저장소의 코드를 클론(Clone)하거나 다운로드합니다.

### 3. 경로 및 파라미터 설정
main.py (또는 해당 코드 파일)을 열어 상단의 전역 설정 및 파라미터 부분을 수정해야 합니다.
```
# --- 1. 전역 설정 및 파라미터 ---
VIDEO_PATH = "/home/nas/data/YeonSeung/chicken_7min.mp4" # 분석할 비디오 경로 (필수 수정)
MODEL_PATH = "/home/a202192020/객체탐지 프로젝트(송교수님)/양륜비박사님모델/weights/best.pt" # YOLO 모델 경로 (필수 수정)
BASE_DIR = "/home/a202192020/객체탐지 프로젝트(송교수님)/chicken_proj" # 결과 저장 기본 디렉토리 (필수 수정)

# 분석 파라미터 (필요에 따라 조정)
BG_LEARNING_RATE = 0.01          # 배경 모델 학습률
STATIC_PIXEL_THRESHOLD = 8       # 픽셀 변화 임계값 (정적 픽셀 판단 기준)
MIN_STATIC_AREA = 200            # 최소 정적 영역 크기 (노이즈 필터링)
YOLO_CONF_THRESHOLD = 0.3        # YOLO 탐지 신뢰도 임계값
YOLO_NMS_THRESHOLD = 0.5         # YOLO NMS 임계값
IOU_TRACKING_THRESHOLD = 0.3     # 트래킹을 위한 IoU 임계값
STATIC_RATIO_THRESHOLD = 0.7     # 바운딩 박스 내 정적 픽셀 비율 임계값
CONSECUTIVE_STATIC_FRAMES = 90   # 닭을 '정적'으로 판정하기 위한 연속 프레임 수 (예: 3초 = 30FPS * 3)
DEAD_CHICKEN_MIN_DURATION_MINUTES = 1 # '죽은 닭'으로 최종 판정할 최소 정적 시간 (분)
```
- VIDEO_PATH: 분석하고자 하는 동영상 파일의 절대 경로를 입력하세요.
- MODEL_PATH: 미리 학습된 닭 탐지 YOLOv8 모델 파일(best.pt와 같은 .pt 확장자)의 절대 경로를 입력하세요.
- BASE_DIR: 분석 결과(생성될 동영상 파일)가 저장될 상위 디렉토리의 절대 경로를 입력하세요. 실행 시각을 기준으로 하위 폴더가 자동으로 생성됩니다.
- 나머지 분석 파라미터들은 닭의 정적/죽음 판정 기준 및 탐지 성능에 영향을 미치므로, 필요에 따라 조정할 수 있습니다.
### 4. GPU 사용 (선택 사항)
만약 NVIDIA GPU가 장착된 시스템이라면, YOLO 모델이 GPU를 사용하여 더 빠르게 동작할 수 있어요. 별도의 설정이 없다면 device=0으로 설정되어 첫 번째 GPU를 사용하려고 시도하며, GPU가 없으면 자동으로 CPU를 사용합니다.
### 5. 실행
실행하시면 됩니다.

## 💻 코드 설명 및 흐름
이 코드는 다음과 같은 흐름으로 동작해요:

### 1. 초기 설정 및 자원 로드
전역 파라미터 정의: 동영상 경로, 모델 경로, 저장 경로, 그리고 분석에 사용될 다양한 임계값들이 정의됩니다.
결과 저장 디렉토리 생성: 스크립트 실행 시각(RUN_ID)을 기반으로 고유한 결과 폴더(SAVE_DIR)가 생성됩니다.
원본 비디오 정보 로드: cv2.VideoCapture를 사용하여 분석할 동영상의 너비, 높이, FPS(초당 프레임 수), 총 프레임 수 등의 정보를 얻습니다.
결과 비디오 객체 생성: 분석 결과를 저장할 새로운 동영상 파일(test_two_segments_overlay_video.mp4)을 위한 cv2.VideoWriter 객체를 생성합니다.
YOLO 모델 로드: 지정된 경로에서 YOLOv8 모델을 메모리에 로드합니다.
배경 서브트랙터 초기화: 각 분석 대상 세그먼트(왼쪽 가운데, 오른쪽 가장 위)에 대해 독립적인 cv2.createBackgroundSubtractorMOG2 객체를 초기화합니다. 이는 각 영역의 배경 변화를 개별적으로 감지하기 위함이에요.
트래킹 및 상태 관리 변수 초기화: 각 세그먼트별로 닭의 트래킹 정보(segment_tracks), 다음 부여할 닭 ID(next_chicken_id_in_segment), 그리고 '죽은 닭' 상태를 저장하는 딕셔너리(dead_chickens_status)를 독립적으로 초기화합니다.
### 2. 핵심 유틸리티 함수
calculate_iou(box1, box2): 두 개의 바운딩 박스 간의 IoU(Intersection over Union) 값을 계산하는 함수예요. 이는 닭 트래킹 시 이전 프레임의 닭과 현재 프레임의 닭이 동일한 개체인지 판단하는 데 사용됩니다.
### 3. 세그먼트 분석 및 트래킹 함수 (analyze_segment_frame)
이 함수는 단일 동영상 프레임의 특정 세그먼트(segment_frame)를 입력받아 분석을 수행합니다.
배경 서브트랙션: bg_subtractor를 사용하여 segment_frame에서 움직이는 객체를 제외한 배경 마스크와, 프레임 간의 픽셀 변화를 통해 정적 마스크를 생성해요. MIN_STATIC_AREA를 사용하여 작은 노이즈 영역은 제거합니다.
YOLO 탐지: YOLO 모델을 사용하여 해당 segment_frame 내에서 닭을 탐지하고, 바운딩 박스 목록을 얻습니다.
닭 트래킹:
현재 프레임에서 탐지된 닭들을 이전에 트래킹하고 있던 닭들과 calculate_iou를 사용하여 매칭합니다.
매칭된 닭은 트래킹 정보를 업데이트하고, 매칭되지 않은 새로운 닭은 새 ID를 부여받아 트래킹 목록에 추가돼요.
이전 프레임에 있었으나 현재 프레임에서 사라진 닭은 트래킹 목록에서 제거됩니다(단, '죽은 닭'으로 판정된 경우는 유지).
정적 상태 및 '죽은 닭' 판정:
각 닭의 바운딩 박스 내에서 정적 마스크의 픽셀 비율(static_ratio)을 계산합니다.
static_ratio가 STATIC_RATIO_THRESHOLD보다 높으면 해당 닭은 '정적 픽셀'이 많다고 판단해요.
consecutive_static 카운터를 사용하여 연속적으로 정적 상태인 프레임 수를 추적하고, CONSECUTIVE_STATIC_FRAMES를 넘으면 is_currently_static을 True로 설정합니다.
total_static_duration을 사용하여 닭이 전체적으로 정지해 있었던 프레임 수를 누적합니다. 이 값이 DEAD_CHICKEN_MIN_DURATION_MINUTES로 설정된 시간을 초과하면 해당 닭은 죽은 닭으로 최종 판정됩니다.
업데이트된 트래킹 정보, 다음 닭 ID, 죽은 닭 상태, 그리고 static_mask를 반환합니다.
### 4. 메인 동영상 처리 루프
tqdm을 사용하여 전체 동영상 처리 진행률을 터미널에 표시해요.
프레임별 처리: 동영상의 각 프레임을 순회하며 읽어옵니다.
두 세그먼트 추출 및 분석:
원본 프레임에서 '왼쪽 가운데' 영역과 '오른쪽 가장 위' 영역을 각각 잘라내어 독립적인 세그먼트 이미지(segment_of_interest1, segment_of_interest2)를 생성합니다.
각 세그먼트 이미지에 대해 analyze_segment_frame 함수를 호출하여 분석을 수행하고 업데이트된 트래킹 및 상태 정보를 받습니다.
결과 시각화 오버레이:
원본 original_frame의 복사본인 display_frame에 다음 요소들을 그려요.
세그먼트 경계선: 각 ROI를 구분하기 위해 노란색(왼쪽 가운데)과 마젠타색(오른쪽 가장 위)의 테두리를 그립니다.
정적 마스크: 각 세그먼트 내에서 정적이라고 판단된 영역에 파란색 반투명 오버레이를 추가합니다.
닭 바운딩 박스: 각 닭의 바운딩 박스를 그리고 ID와 현재 상태(움직임, 정적, 죽음)를 텍스트로 표시합니다. 닭의 상태에 따라 바운딩 박스와 텍스트의 색상이 변경돼요.
전체 상태 텍스트: 비디오 상단에 현재 프레임 번호와 각 세그먼트별 정적 닭 수, 죽은 닭 수를 표시합니다.
결과 비디오 저장: display_frame을 out(cv2.VideoWriter) 객체에 기록하여 최종 결과 동영상 파일을 생성합니다.
prev_gray_segment 변수를 현재 프레임의 회색 이미지를 복사하여 다음 프레임 분석에 사용하도록 업데이트합니다.
### 5. 종료
모든 프레임 처리가 완료되면 cap_orig (원본 비디오)와 out (결과 비디오) 객체를 해제하고, 총 분석 시간을 출력합니다.

## 🎬 결과물
이 스크립트를 성공적으로 실행하면, BASE_DIR 아래에 two_segments_test_YYYYMMDD_HHMMSS와 같은 이름의 새로운 폴더가 생성돼요. 이 폴더 안에 다음 파일이 저장됩니다:

test_two_segments_overlay_video.mp4:
원본 동영상에 왼쪽 가운데 영역과 오른쪽 가장 위 영역의 닭 분석 결과(바운딩 박스, 상태, 정적 마스크)가 실시간으로 오버레이된 비디오 파일입니다.
비디오를 통해 닭들이 특정 영역에서 움직이는지, 정지해 있는지, 혹은 '죽은 닭'으로 판정되었는지 시각적으로 확인할 수 있어요.
각 세그먼트의 경계선과 해당 세그먼트의 닭 ID도 함께 표시됩니다.
