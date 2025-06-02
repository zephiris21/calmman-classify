#!/usr/bin/env python3
"""
CPU 환경 전용 설치 검증 스크립트
"""

print("=== CPU 환경 패키지 검증 ===")

# 필수 패키지들
packages = [
    'torch', 'torchvision', 'timm', 'cv2', 'PIL', 
    'numpy', 'sklearn', 'matplotlib', 'seaborn',
    'albumentations', 'yaml', 'tqdm', 'psutil'
]

failed_packages = []

for pkg in packages:
    try:
        __import__(pkg)
        print(f"✅ {pkg}")
    except ImportError:
        print(f"❌ {pkg}")
        failed_packages.append(pkg)

# PyTorch CPU 테스트
print(f"\n🖥️ PyTorch CPU 정보:")
import torch
print(f"   버전: {torch.__version__}")
print(f"   CUDA 사용 가능: {torch.cuda.is_available()}")
print(f"   CPU 스레드 수: {torch.get_num_threads()}")

# CPU 연산 성능 테스트
import time
print(f"\n⚡ CPU 성능 테스트:")
start_time = time.time()
x = torch.randn(1000, 1000)
y = torch.randn(1000, 1000)
z = x @ y
cpu_time = time.time() - start_time
print(f"   1000x1000 행렬곱: {cpu_time:.3f}초")

# 시스템 정보
import psutil
print(f"\n💻 시스템 정보:")
print(f"   CPU 코어: {psutil.cpu_count()}")
print(f"   메모리: {psutil.virtual_memory().total / (1024**3):.1f} GB")
print(f"   CPU 사용률: {psutil.cpu_percent(interval=1):.1f}%")

# FaceNet CPU 테스트
try:
    from facenet_pytorch import MTCNN
    # CPU 디바이스로 강제 설정
    mtcnn = MTCNN(device='cpu')
    print(f"✅ FaceNet MTCNN (CPU 모드)")
except ImportError as e:
    print(f"❌ FaceNet 문제: {e}")
    failed_packages.append('facenet-pytorch')

# Timm 모델 테스트
try:
    import timm
    # CPU에서 EfficientNet 로드 테스트
    model = timm.create_model('efficientnet_b0', pretrained=False)
    print(f"✅ Timm EfficientNet (CPU 모드)")
except Exception as e:
    print(f"❌ Timm 문제: {e}")

# 최종 결과
if failed_packages:
    print(f"\n❌ 설치 실패 패키지: {failed_packages}")
    print(f"재설치 명령: pip install {' '.join(failed_packages)}")
else:
    print(f"\n🎉 모든 패키지가 CPU 환경에서 정상 작동합니다!")
    print(f"💡 CPU 모드 특징:")
    print(f"   - 처리 속도: GPU 대비 5-10배 느림")
    print(f"   - 메모리 사용량: 더 적음")
    print(f"   - 안정성: 높음")