# CSV 로거 수정 - 분할 영역 정보 추가
import csv
import os

class CSVLogger:
    def __init__(self, save_dir):
        self.csv_path = os.path.join(save_dir, "dead_chickens.csv")
        self.id_mapping_path = os.path.join(save_dir, "id_mapping.csv")

        # 초기 헤더 작성 - part_name 열 추가
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "id", "cx", "cy", "part_name"])

        with open(self.id_mapping_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["new_id", "old_id"])

    def log_dead(self, frame_idx, dead_dict):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            
            # 필요한 경우 구조 변환 (cx, cy) 또는 (cx, cy, part_name)
            for id_, info in dead_dict.items():
                if isinstance(info, tuple):
                    if len(info) >= 3:
                        cx, cy, part_name = info
                    else:
                        cx, cy = info
                        part_name = "unknown"
                else:
                    print(f"경고: ID {id_}의 정보가 예상치 못한 형식입니다: {info}")
                    continue
                    
                writer.writerow([frame_idx, id_, round(cx, 2), round(cy, 2), part_name])

    def log_id_mapping(self, mapping_dict):
        with open(self.id_mapping_path, "a", newline="") as f:
            writer = csv.writer(f)
            for new_id, old_id in mapping_dict.items():
                writer.writerow([new_id, old_id])