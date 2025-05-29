import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path
import numpy as np

def setup(rank, world_size):
    """분산 처리 초기화"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 분산 백엔드 초기화
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # 현재 프로세스의 GPU 설정
    torch.cuda.set_device(rank)
    
    print(f"[rank {rank}] 설정 완료")

def cleanup():
    """분산 처리 종료"""
    dist.destroy_process_group()

def process_video_chunk(rank, world_size, args):
    """각 GPU에서 비디오의 일부를 처리"""
    from main_detector import process_video_segment  # 기본 감지 코드 임포트
    
    # 분산 환경 설정
    setup(rank, world_size)
    
    # 비디오 파일 정보 가져오기
    import cv2
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise ValueError(f"비디오를 열 수 없습니다: {args.video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # 각 GPU에 할당할 프레임 범위 계산
    chunk_size = total_frames // world_size
    start_frame = rank * chunk_size
    end_frame = start_frame + chunk_size if rank < world_size - 1 else total_frames
    
    print(f"[rank {rank}] 프레임 범위: {start_frame} ~ {end_frame-1}")
    
    # 해당 프레임 범위에 대한 비디오 처리
    result_dir = os.path.join(args.base_dir, f"dead_check_{args.run_id}_gpu{rank}")
    os.makedirs(result_dir, exist_ok=True)
    
    # 기본 감지 코드 호출하여 비디오 세그먼트 처리
    process_video_segment(
        video_path=args.video_path,
        model_path=args.model_path,
        save_dir=result_dir,
        start_frame=start_frame,
        end_frame=end_frame,
        device=rank
    )
    
    # 동기화 대기
    dist.barrier()
    
    # 분산 환경 정리
    cleanup()

def merge_results(base_dir, run_id, world_size):
    """각 GPU의 결과 병합"""
    import pandas as pd
    import matplotlib.pyplot as plt
    import cv2
    from glob import glob
    
    # 메인 결과 디렉토리
    main_dir = os.path.join(base_dir, f"dead_check_{run_id}")
    os.makedirs(main_dir, exist_ok=True)
    
    # 로그 병합
    all_logs = []
    for rank in range(world_size):
        gpu_dir = os.path.join(base_dir, f"dead_check_{run_id}_gpu{rank}")
        log_path = os.path.join(gpu_dir, "detection_log.csv")
        
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
            all_logs.append(df)
    
    if all_logs:
        merged_log = pd.concat(all_logs, ignore_index=True)
        merged_log.sort_values(by='frame', inplace=True)
        merged_log.to_csv(os.path.join(main_dir, "detection_log.csv"), index=False)
        
        # 통합 그래프 생성
        plt.figure(figsize=(15, 8))
        for chicken_id in merged_log['chicken_id'].unique():
            chicken_data = merged_log[merged_log['chicken_id'] == chicken_id]
            plt.plot(chicken_data['frame'], chicken_data['is_static'], 
                     label=f'닭 ID: {chicken_id}')
        
        plt.xlabel('프레임')
        plt.ylabel('정적 상태 (1=정적, 0=움직임)')
        plt.title('닭 정적 상태 변화 (통합)')
        plt.grid(True)
        plt.legend()
        
        plt.savefig(os.path.join(main_dir, 'static_status_graph.png'))
    
    # 비디오 병합
    # GPU별 비디오 경로 목록
    video_files = []
    for rank in range(world_size):
        gpu_dir = os.path.join(base_dir, f"dead_check_{run_id}_gpu{rank}")
        video_path = os.path.join(gpu_dir, "output.mp4")
        if os.path.exists(video_path):
            video_files.append((rank, video_path))
    
    # 비디오 정렬 및 병합
    if video_files:
        video_files.sort(key=lambda x: x[0])  # 랭크 순서대로 정렬
        
        # 첫 번째 비디오에서 속성 가져오기
        first_video = cv2.VideoCapture(video_files[0][1])
        width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = first_video.get(cv2.CAP_PROP_FPS)
        first_video.release()
        
        # 병합 비디오 설정
        merged_video_path = os.path.join(main_dir, "output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(merged_video_path, fourcc, fps, (width, height))
        
        # 각 비디오 순차적으로 병합
        for _, video_path in video_files:
            cap = cv2.VideoCapture(video_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            cap.release()
        
        out.release()
    
    # 보고서 병합
    all_reports = {}
    for rank in range(world_size):
        gpu_dir = os.path.join(base_dir, f"dead_check_{run_id}_gpu{rank}")
        report_path = os.path.join(gpu_dir, "report.txt")
        
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report_text = f.read()
                all_reports[rank] = report_text
    
    # 통합 보고서 작성
    merged_report_path = os.path.join(main_dir, "report.txt")
    with open(merged_report_path, 'w') as f:
        f.write(f"닭 움직임 분석 통합 보고서\n")
        f.write(f"=======================\n\n")
        
        # 각 GPU의 보고서 내용 추가
        for rank, report in all_reports.items():
            f.write(f"GPU {rank} 보고서:\n")
            f.write("-" * 20 + "\n")
            f.write(report)
            f.write("\n\n")
    
    print(f"결과 병합 완료. 통합 결과는 {main_dir}에 저장되었습니다.")

def run_multi_gpu(video_path, model_path, base_dir):
    """A100 4장을 활용한 병렬 처리 메인 함수"""
    # 실행 ID 생성
    from datetime import datetime
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 인자 준비
    class Args:
        def __init__(self):
            self.video_path = video_path
            self.model_path = model_path
            self.base_dir = base_dir
            self.run_id = run_id
    
    args = Args()
    
    # GPU 수 확인
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise ValueError("GPU를 찾을 수 없습니다.")
    
    print(f"감지된 GPU 수: {world_size}")
    print(f"실행 ID: {run_id}")
    
    # 멀티프로세싱 시작
    mp.spawn(
        process_video_chunk,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )
    
    # 결과 병합
    print("모든 GPU 처리 완료. 결과 병합 중...")
    merge_results(base_dir, run_id, world_size)
    
    return os.path.join(base_dir, f"dead_check_{run_id}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='멀티 GPU 닭 움직임 감지')
    parser.add_argument('--video', required=True, help='비디오 파일 경로')
    parser.add_argument('--model', required=True, help='YOLO 모델 경로')
    parser.add_argument('--output', required=True, help='결과 저장 기본 경로')
    
    args = parser.parse_args()
    
    result_dir = run_multi_gpu(args.video, args.model, args.output)
    print(f"처리 완료. 결과 위치: {result_dir}")