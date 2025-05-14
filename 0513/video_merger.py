import cv2
import numpy as np
from datetime import datetime

def merge_and_draw(frames_dict, dead_chickens, frame_idx, save_writer):
    """
    병합된 영상 생성 + 박싱 + 선 + ID + 정보 표시
    각 분할 영역별 좌표를 전체 영상 내 절대 좌표로 변환하여 박싱
    
    Args:
        frames_dict: 각 분할 영상 프레임 딕셔너리
        dead_chickens: 움직이지 않는 닭 정보 {id: (cx, cy, part_name)}
        frame_idx: 현재 프레임 인덱스
        save_writer: 영상 저장 객체
    """
    # 분할 영상의 크기 확인
    sample_frame = next(iter(frames_dict.values()))
    part_h, part_w, _ = sample_frame.shape
    
    # 그리드 생성
    grid = []
    for i in range(3):
        row = []
        for j in range(3):
            key = f"part_{i}_{j}"
            frame = frames_dict.get(key, np.zeros((part_h, part_w, 3), dtype=np.uint8))
            
            # 각 분할 영상에 영역 표시
            cv2.putText(frame, key, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
            
            row.append(frame)
        grid.append(np.hstack(row))
    full_frame = np.vstack(grid)

    # 전체 합친 영상의 크기
    h, w, _ = full_frame.shape

    # 9분할 선 그리기
    for i in range(1, 3):
        cv2.line(full_frame, (0, i * h // 3), (w, i * h // 3), (255, 255, 255), 2)
        cv2.line(full_frame, (i * w // 3, 0), (i * w // 3, h), (255, 255, 255), 2)

    # 정지한 닭 박싱
    for id_, chicken_info in dead_chickens.items():
        # chicken_info가 튜플인지 확인하고 적절하게 처리
        if isinstance(chicken_info, tuple):
            if len(chicken_info) >= 3:
                cx, cy, part_name = chicken_info
            else:
                cx, cy = chicken_info[:2]
                part_name = None
        else:
            # 예상치 못한 형식이면 다음 항목으로 넘어감
            print(f"경고: ID {id_}의 닭 정보가 예상치 못한 형식입니다: {chicken_info}")
            continue
            
        # 분할 영역 정보로부터 전체 영상에서의 좌표 계산
        if part_name:
            parts = part_name.split('_')
            if len(parts) >= 3:
                part_i = int(parts[1])
                part_j = int(parts[2])
                
                # 전체 영상에서의 절대 좌표 계산
                abs_cx = int(cx + part_j * part_w)
                abs_cy = int(cy + part_i * part_h)
                
                # 디버그 출력
                print(f"ID {id_}: 원본 좌표 ({cx}, {cy}), 분할 영역 {part_name}, 절대 좌표 ({abs_cx}, {abs_cy})")
            else:
                # 분할 영역 정보 형식이 잘못된 경우
                abs_cx, abs_cy = int(cx), int(cy)
                print(f"경고: ID {id_}의 분할 영역 정보가 잘못되었습니다: {part_name}")
        else:
            # 분할 영역 정보가 없는 경우, 원본 좌표 사용
            abs_cx, abs_cy = int(cx), int(cy)
        
        # 바운딩 박스 그리기
        box_size = 20  # 박스 크기
        x1, y1 = max(0, abs_cx - box_size), max(0, abs_cy - box_size)
        x2, y2 = min(w - 1, abs_cx + box_size), min(h - 1, abs_cy + box_size)
        
        # 좌표가 유효한지 확인
        if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
            cv2.rectangle(full_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # ID 표시
            label = f"ID {id_}"
            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(full_frame, (x1, y1 - h_text - 10), (x1 + w_text + 5, y1), (0, 0, 0), -1)
            cv2.putText(full_frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            print(f"경고: ID {id_}의 좌표가 영상 범위를 벗어났습니다: ({x1}, {y1}), ({x2}, {y2})")

    # 정보 표시 영역 (하단)
    info_height = 30
    info_area = np.zeros((info_height, w, 3), dtype=np.uint8)
    
    # 현재 시간 및 프레임 정보
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    info_text = f"프레임: {frame_idx} | 정지 닭: {len(dead_chickens)} | 시간: {current_time}"
    cv2.putText(info_area, info_text, (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 정보 영역을 메인 프레임에 붙이기
    full_frame_with_info = np.vstack([full_frame, info_area])
    
    # 최종 프레임 저장
    save_writer.write(full_frame)