#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
from pathlib import Path
from mtcnn_wrapper import FaceDetector

def process_images(input_dir):
    """
    간단한 이미지 처리 함수 (눈 정렬 포함)
   
    Args:
        input_dir (str): 입력 이미지 디렉토리 경로
    """
    # 프로젝트 루트 디렉토리
    base_dir = Path(__file__).parent.parent.parent
   
    # 출력 디렉토리 (고정)
    output_dir = os.path.join(base_dir, 'data', 'processed', 'new')
    os.makedirs(output_dir, exist_ok=True)
   
    print(f"입력 디렉토리: {input_dir}")
    print(f"출력 디렉토리: {output_dir}")
   
    # 기본 FaceDetector 초기화 (정렬 기능 추가)
    detector = FaceDetector(
        image_size=224,       # 고정 이미지 크기
        margin=20,           # 얼굴 주변 여백
        prob_threshold=0.9,  # 얼굴 감지 확률 임계값
        align_faces=True     # 👁️ 눈 정렬 활성화
    )
   
    # 디렉토리 내 모든 이미지 처리
    processed_count = detector.process_directory(input_dir, output_dir, recursive=True)
   
    print(f"처리 완료! 총 {processed_count}개 이미지에서 얼굴을 감지하여 저장했습니다.")
    print(f"결과 이미지는 {output_dir} 에 저장되었습니다.")
    print("✅ 모든 얼굴이 눈 위치 기준으로 정렬되었습니다.")

def main():
    # 기본 입력 디렉토리 설정
    default_input_dir = r"D:\my_projects\calmman-facial-classification\data\input"
    
    # 인자 파싱
    parser = argparse.ArgumentParser(description='간단한 얼굴 이미지 처리 스크립트 (눈 정렬 포함)')
    parser.add_argument('input_dir', type=str, nargs='?', default=default_input_dir,
                       help=f'처리할 이미지가 있는 디렉토리 경로 (기본값: {default_input_dir})')
   
    args = parser.parse_args()
   
    # 입력 디렉토리 경로 확인
    if not os.path.exists(args.input_dir):
        print(f"오류: 입력 디렉토리 '{args.input_dir}'가 존재하지 않습니다.")
        return
   
    # 이미지 처리 실행
    process_images(args.input_dir)

if __name__ == "__main__":
    main()