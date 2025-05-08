import os
import cv2
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime

# =========================
# 설정값
# =========================
VIDEO_PATH = "비디오파일경로입력"
MODEL_PATH = "모델파일경로입력"
BASE_DIR = "결과 저장되는경로"
FRAME_INTERVAL_SEC = 60
FRAME_BATCH = 100
TOLERANCE_RADIUS = 3  # 중앙점 기준 반지름 (지름 = 6픽셀 허용)

# 결과 저장 폴더
run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
SAVE_DIR = os.path.join(BASE_DIR, f"dead_check_{run_id}")
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# 초기 설정
# =========================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = int(total_frames / fps)

# 4분할
regions = {
    "topleft": (0, width // 2, 0, height // 2),
    "topright": (width // 2, width, 0, height // 2),
    "bottomleft": (0, width // 2, height // 2, height),
    "bottomright": (width // 2, width, height // 2, height)
}

# 중앙점 거리 비교 함수
def is_within_radius(p1, p2, r=TOLERANCE_RADIUS):
    return abs(p1[0] - p2[0]) <= r and abs(p1[1] - p2[1]) <= r

# =========================
# 분석 시작
# =========================
global_id_counter = 0
total_steps = len(regions) * ((duration - 1) // FRAME_INTERVAL_SEC)
current_step = 0

for region_name, (x1, x2, y1, y2) in regions.items():
    print(f"\n--- [{region_name.upper()}] 분석 시작 ---")
    region_dir = os.path.join(SAVE_DIR, region_name)
    os.makedirs(region_dir, exist_ok=True)
    tracking_log = []
    dead_candidate_dict = defaultdict(list)

    for t in range(0, duration - 1, FRAME_INTERVAL_SEC):
        frame_centers = []

        for i in range(FRAME_BATCH):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((t + i / fps) * fps))
            ret, frame = cap.read()
            if not ret:
                continue
            sub = frame[y1:y2, x1:x2]
            results = model.predict(sub, verbose=False)[0]
            boxes = results.boxes.xyxy.cpu().numpy()

            frame_center_set = []
            for box in boxes:
                x_min, y_min, x_max, y_max = box[:4]
                cx = int((x_min + x_max) / 2)
                cy = int((y_min + y_max) / 2)
                frame_center_set.append((cx, cy))
            frame_centers.append(frame_center_set)

        # "거의 안 움직이는" 중심점 찾기
        stable_centers = []
        base_frame = frame_centers[0]
        for base_c in base_frame:
            if all(any(is_within_radius(base_c, other_c) for other_c in centers)
                   for centers in frame_centers[1:]):
                stable_centers.append(base_c)

        # 기존 ID와 매칭 or 신규 ID 부여
        matched_ids = set()
        for center in stable_centers:
            matched = False
            for id_, history in dead_candidate_dict.items():
                last_frame, last_center = history[-1]
                if abs(t * fps - last_frame) >= FRAME_INTERVAL_SEC * fps:
                    if is_within_radius(center, last_center):
                        for offset in range(FRAME_BATCH):
                            frame_index = int(round((t + offset / fps) * fps))
                            dead_candidate_dict[id_].append((frame_index, center))
                        matched = True
                        matched_ids.add(id_)
                        break
            if not matched:
                for offset in range(FRAME_BATCH):
                    frame_index = int(round((t + offset / fps) * fps))
                    dead_candidate_dict[global_id_counter].append((frame_index, center))
                matched_ids.add(global_id_counter)
                global_id_counter += 1

        # 움직인 ID는 제거
        for id_ in list(dead_candidate_dict.keys()):
            if id_ not in matched_ids:
                del dead_candidate_dict[id_]

        print(f"[{region_name}] {t//60}분 분석 완료 | 현재 죽은 후보 ID: {list(dead_candidate_dict.keys())}")

    # 로그 저장
    for id_, hist in dead_candidate_dict.items():
        for fidx, pt in hist:
            tracking_log.append({
                "id": id_,
                "frame": fidx,
                "x": pt[0],
                "y": pt[1]
            })

    df = pd.DataFrame(tracking_log)
    df.to_csv(os.path.join(region_dir, f"{region_name}_dead_candidates.csv"), index=False)

cap.release()
print(f"\n분석 완료, 결과는 {SAVE_DIR} 에 저장됨.")
