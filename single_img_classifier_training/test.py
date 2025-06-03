# test_model_structure.py
import torch

# 1. 저장된 모델 구조 확인
model_path = "results/affectnet_simple/models/best_model_stage2_20250603_185221.pth"
checkpoint = torch.load(model_path, weights_only=False)

print("=== 저장된 모델 구조 분석 ===")
print(f"타입: {type(checkpoint)}")

if isinstance(checkpoint, dict):
    print("\n🔍 state_dict 키들:")
    for i, key in enumerate(checkpoint.keys()):
        print(f"  {i:2d}: {key}")
        if i > 20:  # 처음 20개만
            print("     ... (더 많은 키들)")
            break
    
    # backbone 구조 추론
    backbone_keys = [k for k in checkpoint.keys() if k.startswith('backbone.')]
    print(f"\n🏗️ backbone 관련 키 개수: {len(backbone_keys)}")
    print("backbone 키들:")
    for key in backbone_keys[:10]:
        print(f"  {key}")
    
    # classifier 구조 확인
    classifier_keys = [k for k in checkpoint.keys() if k.startswith('classifier.')]
    print(f"\n📋 classifier 관련 키들:")
    for key in classifier_keys:
        print(f"  {key}: {checkpoint[key].shape}")

else:
    print("전체 모델이 저장됨")
    print(f"모델 구조: {checkpoint}")