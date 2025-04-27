import os
import cv2
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO
import pandas as pd

# 모델 로드
model_path = "/home/a202192020/객체탐지 프로젝트(송교수님)/양륜비박사님모델/weights/best.pt"
model = YOLO(model_path)

# 결과 저장할 폴더 자동 생성
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
base_output_dir = f"/home/a202192020/객체탐지 프로젝트(송교수님)/chicken_proj/run_{timestamp}"
os.makedirs(base_output_dir, exist_ok=True)
video_output_path = os.path.join(base_output_dir, "result.mp4")
csv_output_path = os.path.join(base_output_dir, "result.csv")

# 영상 로드
video_path = "/home/nas/data/YeonSeung/0_8_IPC1_20230108162038.mp4"
cap = cv2.VideoCapture(video_path)

# 영상 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

# ID 별 좌표 기록, 마지막 움직인 프레임 기록
id_positions = defaultdict(list)
id_last_movement_frame = dict()
id_dead_probability = dict()

# CSV 저장용 데이터
csv_data = []

frame_idx = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 총 프레임 수 읽기

# 메인 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, tracker="bytetrack.yaml")
    boxes = results[0].boxes

    if boxes is not None and boxes.id is not None:
        for box, track_id in zip(boxes.xyxy.cpu().numpy(), boxes.id.cpu().numpy()):
            track_id = int(track_id)
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # 기존 좌표랑 비교해서 이동 여부 판단
            if id_positions[track_id]:
                prev_cx, prev_cy = id_positions[track_id][-1]
                if np.hypot(cx - prev_cx, cy - prev_cy) > 5:  # 이동했으면
                    id_last_movement_frame[track_id] = frame_idx
            else:
                id_last_movement_frame[track_id] = frame_idx

            id_positions[track_id].append((cx, cy))

            # 현재 프레임에서 마지막으로 움직인 프레임까지 차이 계산
            no_movement_frames = frame_idx - id_last_movement_frame.get(track_id, frame_idx)

            # 확률 계산
            if no_movement_frames >= 7200:  # 4분 이상
                dead_prob = 80 + min((no_movement_frames - 7200) / 3600 * 10, 10)
            elif no_movement_frames >= 3600:  # 2분 이상
                dead_prob = 70
            elif no_movement_frames >= 1800:  # 1분 이상
                dead_prob = 50
            elif no_movement_frames >= 900:   # 30초 이상
                dead_prob = 40
            else:
                dead_prob = 0

            id_dead_probability[track_id] = dead_prob

            # 박싱 + 확률 표시
            label = f"ID {track_id}: {dead_prob:.0f}%"
            color = (0, 0, 255) if dead_prob >= 50 else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # CSV 데이터 저장
            csv_data.append([frame_idx, track_id, cx, cy, dead_prob])

    out.write(frame)

    # 프레임마다 진행상황 출력
    progress = (frame_idx / total_frames) * 100
    print(f"[{frame_idx}/{total_frames}] ({progress:.2f}%) 프레임 처리 완료.")

    frame_idx += 1

cap.release()
out.release()

# CSV 파일로 저장
df = pd.DataFrame(csv_data, columns=["frame", "id", "center_x", "center_y", "dead_probability"])
df.to_csv(csv_output_path, index=False)

print("\u2705 영상 + CSV 저장 완료:", base_output_dir)
