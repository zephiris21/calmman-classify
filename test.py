import torch
import torch.nn as nn
import sys
import os
from pprint import pprint

def inspect_model(model_path):
    print(f"=== 모델 파일 정보 검사: {model_path} ===\n")
    
    # 기본 파일 정보
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"파일 크기: {file_size_mb:.2f} MB")
    
    try:
        # 모델 로드 시도
        print("\n[1] 모델 로드 시도 (weights_only=False)...")
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        
        # 모델 기본 정보
        print(f"\n모델 타입: {type(model)}")
        print(f"모델 클래스: {model.__class__.__name__}")
        
        # 모델 구조 정보
        print("\n모델 주요 속성:")
        for attr in dir(model):
            if not attr.startswith("_") and not callable(getattr(model, attr)):
                print(f"  - {attr}")
        
        # classifier 확인
        if hasattr(model, 'classifier'):
            classifier = model.classifier
            print(f"\nclassifier 정보: {classifier}")
            print(f"classifier 타입: {type(classifier)}")
            if isinstance(classifier, nn.Module):
                print(f"classifier 구조:\n{classifier}")
        else:
            print("\nclassifier 속성 없음")
        
        # children 구조 확인
        print("\n모델 최상위 계층 구조:")
        for i, child in enumerate(model.children()):
            print(f"  레이어 {i}: {type(child).__name__}")
            if i < 1:  # 첫 번째 레이어 세부 정보만 표시
                print(f"    구조: {child}")
        
        # state_dict 키 확인
        print("\n[2] state_dict 키 샘플 (최대 10개):")
        state_dict = model.state_dict()
        for i, (key, value) in enumerate(list(state_dict.items())[:10]):
            print(f"  - {key}: shape {value.shape}")
        
        print(f"\n총 state_dict 키 개수: {len(state_dict)}")
        
        # 백본 구조 (특별히 확인)
        if hasattr(model, 'backbone'):
            print("\n백본 정보:")
            print(f"  타입: {type(model.backbone)}")
            print(f"  구조: {model.backbone}")
        elif hasattr(model, 'features'):
            print("\nfeatures 정보 (백본 역할):")
            print(f"  타입: {type(model.features)}")
            print(f"  첫 번째 레이어: {list(model.features.children())[0]}")
        
    except Exception as e:
        print(f"\n❌ 모델 로드 오류: {e}")
    
    print("\n=== 검사 완료 ===")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "./models/affectnet_emotions/enet_b2_8.pt"
    
    inspect_model(model_path)