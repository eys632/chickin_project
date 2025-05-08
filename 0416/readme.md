🐔 Chicken Death Detection using YOLOv8 (detect_and_track3Letgo.py)
본 프로젝트는 닭 사육장에서 촬영한 영상을 분석하여, YOLOv8 모델로 객체 탐지한 닭 중 장시간 움직이지 않는 닭을 '죽은 닭 후보'로 자동 식별하는 시스템입니다.
특히, YOLO의 박스 흔들림(노이즈)으로 인해 움직인 것으로 오인하는 문제를 방지하기 위한 로직이 구현되어 있습니다.

📂 파일 개요
파일명: detect_and_track3Letgo.py

사용 모델: Ultralytics YOLOv8

주요 기능:

영상 4분할 후 독립적 분석

닭 중심 좌표 추출 및 안정성 검증

'거의 움직이지 않는 닭' 탐지

사망 가능성 높은 ID 식별 및 로그 저장 (CSV)

📌 주요 아이디어
YOLO로 탐지된 바운딩 박스가 작게 흔들리는 것만으로 움직임으로 판단되는 문제를 해결하기 위해,
여러 프레임 동안 중심 좌표가 거의 동일한 닭만 '안 움직이는 닭'으로 판단합니다.

✔️ 중심 좌표가 반지름 3픽셀 이내로만 움직인 경우 → '정지'로 간주

✔️ 해당 중심 좌표가 일정 시간 이상 반복적으로 등장하면 → '죽은 닭 후보'로 등록

🛠️ 실행 전 준비
1. 사전 설치 패키지
bash
복사
편집
pip install ultralytics opencv-python numpy pandas
2. 필수 입력 파일
YOLO 모델 파일: best.pt

분석할 영상 파일: 예: chicken_7min.mp4

🧠 코드 설명 (전체 흐름)
1. 설정값 정의
python
복사
편집
VIDEO_PATH = "..."        # 분석 대상 영상 경로
MODEL_PATH = "..."        # YOLO 모델 경로
BASE_DIR = "..."          # 결과 저장 베이스 경로
FRAME_INTERVAL_SEC = 60   # 몇 초 간격으로 분석할지
FRAME_BATCH = 100         # 몇 프레임씩 묶어 비교할지
TOLERANCE_RADIUS = 3      # 박스 중심 좌표 허용 반경
2. 4분할 처리
python
복사
편집
regions = {
    "topleft": (0, width//2, 0, height//2),
    ...
}
영상을 좌상, 우상, 좌하, 우하 네 구역으로 나누어 각각 독립적으로 분석합니다.
⇒ 처리 속도 향상 & 위치 기반 후처리를 위함

3. 중심 좌표 비교 (중요 핵심 로직)
python
복사
편집
def is_within_radius(p1, p2, r=3):
    return abs(p1[0] - p2[0]) <= r and abs(p1[1] - p2[1]) <= r
YOLO 바운딩 박스의 중심 좌표 cx, cy를 구함

100개의 프레임에서 좌표가 비슷한 경우만 stable_centers에 등록

YOLO 박스가 흔들려도 무시하도록 설계

4. ID 추적 및 사망 후보 판단
python
복사
편집
# 기존 ID와 비교
if is_within_radius(center, last_center):
    dead_candidate_dict[id_].append((frame_idx, center))
ID마다 중심 좌표의 변화 이력을 기록

새로 등장한 중심 좌표는 global_id_counter로 ID 부여

일정 시간 이상 같은 위치에서 반복적으로 등장한 경우 → '죽은 닭 후보'

5. 결과 저장
CSV: 각 영역별로 사망 후보 닭 ID, 프레임 번호, 좌표 저장

경로: BASE_DIR/dead_check_날짜시간/영역명/영역명_dead_candidates.csv

📊 결과 예시 (CSV)
id	frame	x	y
0	720	114	233
0	721	113	232
0	722	114	233
...	...	...	...

해당 ID가 여러 프레임 동안 거의 같은 좌표에 있었던 경우로,
사망 가능성이 높은 개체로 판단됩니다.

🧪 실제 적용 예
카메라로 촬영된 닭 사육장 영상을 자동 분석하여,

특정 위치에 장시간 정지한 닭을 탐지

육안 확인 없이도 사망 여부 의심 구간을 빠르게 파악 가능

💡 개선 가능 아이디어
사망 확률 점수화 (예: 1분 정지 시 30%, 2분 60% 등)

YOLO + 추적기(TraDeS, DeepSort 등) 연동

사망 여부를 영상으로 표시하는 기능 추가

📁 디렉토리 구조 예시
복사
편집
chicken_proj/
├── dead_check_20250508_151523/
│   ├── topleft/
│   │   └── topleft_dead_candidates.csv
│   ├── topright/
│   ├── ...
🙋‍♂️ 문의
담당자: [너의 이름 작성]

학교/소속: 전주대학교 인공지능학과

프로젝트명: 객체탐지 프로젝트 (닭 사망 여부 판별)