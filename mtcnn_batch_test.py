#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import time

def test_mtcnn_batch_support():
    """MTCNN 배치 처리 지원 여부 테스트"""
    
    print("🔍 MTCNN 배치 처리 지원 테스트")
    print("="*50)
    
    # 디바이스 설정
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}")
    
    # MTCNN 초기화
    mtcnn = MTCNN(
        image_size=224,
        margin=20,
        device=device,
        keep_all=True
    )
    
    print(f"\n📋 MTCNN 메서드 목록:")
    methods = [m for m in dir(mtcnn) if not m.startswith('_')]
    for method in methods:
        print(f"  - {method}")
    
    # 테스트 이미지 생성 (3장)
    test_images = []
    for i in range(3):
        # 랜덤 이미지 생성 (224x224 RGB)
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_img = Image.fromarray(img_array)
        test_images.append(pil_img)
    
    print(f"\n🖼️ 테스트 이미지 생성: {len(test_images)}장")
    
    # 1. 단일 이미지 처리 테스트
    print(f"\n🔬 1. 단일 이미지 처리 테스트")
    try:
        start_time = time.time()
        
        single_results = []
        for i, img in enumerate(test_images):
            boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
            single_results.append((boxes, probs, landmarks))
            print(f"  이미지 {i+1}: {boxes is not None}")
        
        single_time = time.time() - start_time
        print(f"  단일 처리 시간: {single_time:.3f}초")
        
    except Exception as e:
        print(f"  ❌ 단일 처리 실패: {e}")
        return
    
    # 2. 배치 처리 테스트 (리스트로)
    print(f"\n🔬 2. 배치 처리 테스트 (리스트)")
    try:
        start_time = time.time()
        
        # 리스트로 여러 이미지 전달
        batch_boxes, batch_probs, batch_landmarks = mtcnn.detect(
            test_images, landmarks=True
        )
        
        batch_time = time.time() - start_time
        print(f"  ✅ 배치 처리 성공!")
        print(f"  배치 처리 시간: {batch_time:.3f}초")
        print(f"  속도 향상: {single_time/batch_time:.2f}배")
        
        # 결과 형태 확인
        print(f"\n📊 배치 결과 형태:")
        print(f"  batch_boxes 타입: {type(batch_boxes)}")
        if batch_boxes is not None:
            print(f"  batch_boxes 길이: {len(batch_boxes)}")
            for i, boxes in enumerate(batch_boxes):
                print(f"    이미지 {i+1}: {boxes is not None}")
        
        return True, batch_time, single_time
        
    except Exception as e:
        print(f"  ❌ 배치 처리 실패: {e}")
        print(f"  오류 타입: {type(e).__name__}")
        
        # 더 자세한 정보
        import traceback
        print(f"  상세 오류:")
        traceback.print_exc()
        
        return False, None, single_time

def test_mtcnn_api_details():
    """MTCNN API 상세 정보 확인"""
    
    print(f"\n🔍 MTCNN.detect() 메서드 상세 정보")
    print("="*50)
    
    mtcnn = MTCNN(device='cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # detect 메서드 시그니처 확인
    import inspect
    sig = inspect.signature(mtcnn.detect)
    print(f"detect 메서드 시그니처:")
    print(f"  {sig}")
    
    # 파라미터 정보
    print(f"\n파라미터 정보:")
    for param_name, param in sig.parameters.items():
        print(f"  {param_name}: {param.annotation} = {param.default}")
    
    # 독스트링 확인
    if mtcnn.detect.__doc__:
        print(f"\n독스트링:")
        print(f"  {mtcnn.detect.__doc__}")

def test_batch_implementation():
    """간단한 배치 구현 테스트"""
    
    print(f"\n🛠️ 간단한 배치 구현 테스트")
    print("="*50)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(device=device, keep_all=True)
    
    # 테스트 이미지 생성
    test_images = []
    for i in range(5):
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_images.append(Image.fromarray(img_array))
    
    # 배치 처리 함수 구현
    def process_batch_manual(images, batch_size=3):
        """수동 배치 처리"""
        all_results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            print(f"  배치 {i//batch_size + 1}: {len(batch)}개 이미지")
            
            batch_results = []
            for img in batch:
                try:
                    boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
                    batch_results.append((boxes, probs, landmarks))
                except Exception as e:
                    print(f"    오류: {e}")
                    batch_results.append((None, None, None))
            
            all_results.extend(batch_results)
        
        return all_results
    
    # 테스트 실행
    start_time = time.time()
    results = process_batch_manual(test_images)
    end_time = time.time()
    
    print(f"  처리 완료: {len(results)}개 결과")
    print(f"  처리 시간: {end_time - start_time:.3f}초")
    
    return results

if __name__ == "__main__":
    print("🚀 MTCNN 배치 처리 테스트 시작\n")
    
    # 1. 기본 배치 지원 테스트
    success, batch_time, single_time = test_mtcnn_batch_support()
    
    # 2. API 상세 정보 확인
    test_mtcnn_api_details()
    
    # 3. 수동 배치 구현 테스트
    test_batch_implementation()
    
    print(f"\n🎯 테스트 결과 요약:")
    if success:
        print(f"  ✅ MTCNN 네이티브 배치 처리 지원됨")
        print(f"  ⚡ 성능 향상: {single_time/batch_time:.2f}배")
    else:
        print(f"  ❌ MTCNN 네이티브 배치 처리 미지원")
        print(f"  💡 수동 배치 구현 필요")
    
    print(f"\n✅ 테스트 완료!")