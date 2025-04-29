# 필수 라이브러리 임포트
import os  # 파일 및 디렉토리 관리
import cv2  # OpenCV: 영상 처리
import torch  # PyTorch: 모델 구동
import numpy as np  # 수치 계산
from datetime import datetime  # 현재 시간 기록용
from collections import defaultdict  # 기본값이 있는 딕셔너리 생성
from ultralytics import YOLO  # YOLO 객체탐지 모델
import pandas as pd  # CSV 저장용

# 학습된 YOLO 모델 불러오기
model_path = "/home/a202192020/객체탐지 프로젝트(송교수님)/양륜비박사님모델/weights/best.pt"
model = YOLO(model_path)

# 결과 저장 경로 생성 (현재 시간 기반으로 고유하게)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
base_output_dir = f"/home/a202192020/객체탐지 프로젝트(송교수님)/chicken_proj/run_{timestamp}"
os.makedirs(base_output_dir, exist_ok=True)  # 이미 존재해도 에러 없이 생성
video_output_path = os.path.join(base_output_dir, "result.mp4")  # 결과 영상 경로
csv_output_path = os.path.join(base_output_dir, "result.csv")  # 결과 CSV 경로

# 분석할 영상 불러오기
video_path = "/home/nas/data/YeonSeung/0_8_IPC1_20230108162038.mp4"
cap = cv2.VideoCapture(video_path)

# 영상 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 비디오 코덱 설정
fps = cap.get(cv2.CAP_PROP_FPS)  # 초당 프레임 수
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 프레임 가로
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 프레임 세로
out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))  # 결과 영상 저장 객체

# 개체별 좌표, 움직임 기록용 딕셔너리 초기화
id_positions = defaultdict(list)  # ID별 중심 좌표 기록
id_last_movement_frame = dict()  # ID별 마지막 움직임 프레임
id_dead_probability = dict()  # ID별 죽었을 확률 저장

# CSV로 저장할 데이터 리스트
csv_data = []

frame_idx = 0  # 현재 프레임 번호
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 전체 프레임 수

# 메인 루프: 영상 끝날 때까지 반복
while cap.isOpened():
    ret, frame = cap.read()  # 프레임 읽기
    if not ret:
        break  # 읽지 못하면 종료

    # YOLO로 객체 추적 수행 (ByteTrack 사용)
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")
    boxes = results[0].boxes  # 결과에서 박스 정보 추출

    # 박스 정보가 있을 경우 처리
    if boxes is not None and boxes.id is not None:
        for box, track_id in zip(boxes.xyxy.cpu().numpy(), boxes.id.cpu().numpy()):
            track_id = int(track_id)  # ID 정수화
            x1, y1, x2, y2 = map(int, box)  # 좌표 정수형 변환
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # 중심 좌표 계산

            # 이전 위치와 비교하여 움직였는지 확인
            if id_positions[track_id]:
                prev_cx, prev_cy = id_positions[track_id][-1]
                if np.hypot(cx - prev_cx, cy - prev_cy) > 5:  # 이동거리 > 5픽셀
                    id_last_movement_frame[track_id] = frame_idx  # 마지막 이동 프레임 갱신
            else:
                id_last_movement_frame[track_id] = frame_idx  # 첫 등장 시 초기화

            id_positions[track_id].append((cx, cy))  # 위치 기록

            # 현재 프레임 기준, 마지막 움직임 이후 경과 프레임 수
            no_movement_frames = frame_idx - id_last_movement_frame.get(track_id, frame_idx)

            # 움직이지 않은 시간에 따라 죽은 확률 계산 (프레임 기준)
            if no_movement_frames >= 7200:  # 4분 이상 (80~90%)
                dead_prob = 80 + min((no_movement_frames - 7200) / 3600 * 10, 10)
            elif no_movement_frames >= 3600:  # 2분 이상
                dead_prob = 70
            elif no_movement_frames >= 1800:  # 1분 이상
                dead_prob = 50
            elif no_movement_frames >= 900:  # 30초 이상
                dead_prob = 40
            else:  # 30초 미만
                dead_prob = 0

            id_dead_probability[track_id] = dead_prob  # 확률 저장

            # 박스와 텍스트 시각화 (확률 50% 이상은 빨간색)
            label = f"ID {track_id}: {dead_prob:.0f}%"
            color = (0, 0, 255) if dead_prob >= 50 else (0, 255, 0)  # 빨간색 or 초록색
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # 박스 그리기
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)  # 텍스트 표시

            # CSV 데이터 누적 저장
            csv_data.append([frame_idx, track_id, cx, cy, dead_prob])

    out.write(frame)  # 결과 프레임 영상 저장

    # 진행 상황 출력
    progress = (frame_idx / total_frames) * 100
    print(f"[{frame_idx}/{total_frames}] ({progress:.2f}%) 프레임 처리 완료.")

    frame_idx += 1  # 다음 프레임으로 이동

# 모든 프레임 처리 완료 → 영상 저장 종료
cap.release()
out.release()

# CSV 저장
df = pd.DataFrame(csv_data, columns=["frame", "id", "center_x", "center_y", "dead_probability"])
df.to_csv(csv_output_path, index=False)

# 최종 완료 메시지 출력
print("✅ 영상 + CSV 저장 완료:", base_output_dir)
