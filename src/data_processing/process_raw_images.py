#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path

from mtcnn_wrapper import FaceDetector

def process_specific_emotions(detector, input_base_dir, output_base_dir):
    """
    침착맨 특유 표정 이미지 처리
    
    Args:
        detector: FaceDetector 인스턴스
        input_base_dir: 입력 기본 디렉토리
        output_base_dir: 출력 기본 디렉토리
    """
    # 클래스 매핑 (디렉토리명 -> 클래스명)
    class_mapping = {
        '기본표정': 'neutral',
        '몰입': 'focused',
        '세모입': 'triangle_mouth',
        '약올리기': None,  # 약올리기는 별도 처리
        '열받음': 'angry',
        '웃김': 'funny',
        '장난스러움': 'playful'
    }
    
    # 출력 디렉토리
    specific_output_dir = os.path.join(output_base_dir, 'specific')
    teasing_output_dir = os.path.join(output_base_dir, 'teasing')
    non_teasing_output_dir = os.path.join(output_base_dir, 'non_teasing')
    
    # 각 클래스별 처리
    for input_class, output_class in class_mapping.items():
        input_dir = os.path.join(input_base_dir, input_class)
        
        # 디렉토리가 없으면 건너뜀
        if not os.path.exists(input_dir):
            print(f"디렉토리가 없음: {input_dir}")
            continue
            
        if input_class == '약올리기':
            # 약올리기 클래스는 teasing 디렉토리로 처리
            print(f"약올리기 이미지 처리 중...")
            detector.process_directory(input_dir, teasing_output_dir)
        else:
            # 약올리기가 아닌 클래스는 specific 디렉토리 내 해당 클래스로 처리
            # 그리고 non_teasing 디렉토리로도 복사
            print(f"{input_class} 이미지 처리 중...")
            
            # specific 디렉토리 처리
            specific_class_dir = os.path.join(specific_output_dir, output_class)
            detector.process_directory(input_dir, specific_class_dir)
            
            # non_teasing 디렉토리로도 처리 (중복 처리됨)
            detector.process_directory(input_dir, non_teasing_output_dir)
    
    print(f"침착맨 특유 표정 이미지 처리 완료")

def main():
    # 인자 파싱
    parser = argparse.ArgumentParser(description='얼굴 이미지 처리 스크립트')
    parser.add_argument('--input-dir', type=str, default=None, 
                        help='입력 이미지 디렉토리 (기본값: data/raw)')
    parser.add_argument('--output-dir', type=str, default=None, 
                        help='출력 이미지 디렉토리 (기본값: data/processed)')
    parser.add_argument('--image-size', type=int, default=224, 
                        help='출력 이미지 크기 (기본값: 224)')
    parser.add_argument('--margin', type=int, default=20, 
                        help='얼굴 주변 여백 (기본값: 20)')
    parser.add_argument('--prob-threshold', type=float, default=0.9, 
                        help='얼굴 감지 확률 임계값 (기본값: 0.9)')
    parser.add_argument('--device', type=str, default=None, 
                        help='사용할 장치 (예: cpu, cuda)')
    
    args = parser.parse_args()
    
    # 기본 경로 설정
    base_dir = Path(__file__).parent.parent.parent  # 프로젝트 루트 디렉토리
    
    # 인자에서 디렉토리 경로 가져오기 (없으면 기본값 사용)
    input_dir = args.input_dir if args.input_dir else os.path.join(base_dir, 'data', 'raw')
    output_dir = args.output_dir if args.output_dir else os.path.join(base_dir, 'data', 'processed')
    
    print(f"입력 디렉토리: {input_dir}")
    print(f"출력 디렉토리: {output_dir}")
    
    # FaceDetector 초기화
    detector = FaceDetector(
        image_size=args.image_size,
        margin=args.margin,
        prob_threshold=args.prob_threshold,
        device=args.device
    )
    
    # 침착맨 특유 표정 처리
    process_specific_emotions(detector, input_dir, output_dir)
    
    print("모든 처리가 완료되었습니다!")

if __name__ == "__main__":
    main() 