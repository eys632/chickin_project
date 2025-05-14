import numpy as np
from collections import defaultdict

class MotionTracker:
    def __init__(self, threshold_frame=1800):  # 1분 기준 (30fps × 60초)
        self.positions = defaultdict(list)     # id: [(frame_idx, (cx, cy), part_name)]
        self.dead_ids = set()
        self.current_id = 0
        self.id_mapping = {}                   # new_id: old_id
        self.coord_history = {}                # old_id: (cx, cy, part_name)

    def update_positions(self, frame_idx, detections, part_name=None):
        """
        현재 프레임의 감지 결과 업데이트
        각 감지 위치에 적절한 ID 할당
        """
        frame_result = {}
        for cx, cy in detections:
            # 분할 영역에서 인덱스 추출
            if part_name:
                parts = part_name.split('_')
                if len(parts) >= 3:
                    part_i = int(parts[1])
                    part_j = int(parts[2])
                else:
                    part_i = 0
                    part_j = 0
            else:
                part_i = 0
                part_j = 0

            # 근처 정적 객체 탐색
            matched_id = self.find_nearby_static(cx, cy, part_i, part_j)

            if matched_id is not None:
                # 기존 객체에 좌표 추가
                self.positions[matched_id].append((frame_idx, (cx, cy), part_name))
                frame_result[matched_id] = (cx, cy)

                # 새 ID로 매핑되지 않았다면 매핑 추가
                if matched_id not in self.id_mapping.values():
                    self.id_mapping[self.current_id] = matched_id
            else:
                # 새 ID 부여
                new_id = self.current_id
                self.positions[new_id].append((frame_idx, (cx, cy), part_name))
                self.coord_history[new_id] = (cx, cy, part_name)
                frame_result[new_id] = (cx, cy)
                self.current_id += 1

        return frame_result

    def find_nearby_static(self, cx, cy, part_i=0, part_j=0, threshold=30):
        """
        좌표 근처에 과거 정적 객체가 있는지 탐색
        """
        best_dist = float('inf')
        best_id = None

        for id_, (px, py, p_name) in self.coord_history.items():
            # 분할 영역이 다르면 무시
            if p_name:
                p_parts = p_name.split('_')
                if len(p_parts) >= 3:
                    p_i = int(p_parts[1])
                    p_j = int(p_parts[2])
                else:
                    p_i = 0
                    p_j = 0

                if part_i != p_i or part_j != p_j:
                    continue

            # 거리 계산
            dist = np.hypot(cx - px, cy - py)
            if dist < threshold and dist < best_dist:
                best_dist = dist
                best_id = id_

        return best_id

    def get_dead_chickens(self, current_frame, tolerance=5):
        """
        현재 프레임 기준으로 움직이지 않는 닭 반환
        """
        dead_list = {}
        for id_, pos_list in self.positions.items():
            if len(pos_list) < 2:
                continue

            recent = [f for f, _, _ in pos_list if current_frame - f <= 1800]
            if len(recent) >= tolerance:
                last_frame, (cx, cy), part_name = pos_list[-1]
                if part_name:
                    parts = part_name.split('_')
                    if len(parts) >= 3:
                        part_i = int(parts[1])
                        part_j = int(parts[2])
                        dead_list[id_] = (cx, cy, part_name)
                else:
                    dead_list[id_] = (cx, cy, None)
                self.dead_ids.add(id_)

        return dead_list
