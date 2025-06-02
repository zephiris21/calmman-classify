#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mac용 PyTorch MPS 테스트 스크립트
이 스크립트는 Apple Silicon Mac에서 PyTorch의 MPS 가속화 지원을 확인합니다.
"""

import torch
import sys
import platform

def print_separator():
    print("-" * 50)

print_separator()
print("📊 시스템 정보:")
print(f"OS: {platform.system()} {platform.release()}")
print(f"Python 버전: {platform.python_version()}")
print(f"프로세서: {platform.processor()}")
print_separator()

print("📦 PyTorch 정보:")
print(f"PyTorch 버전: {torch.__version__}")

# MPS 지원 확인 (Apple Silicon Mac 전용)
print_separator()
print("🔍 MPS 지원 확인:")

if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    mps_available = True
    print("✅ MPS 가속화 지원됨 (Apple Silicon)")
    print(f"   MPS 빌드 여부: {torch.backends.mps.is_built()}")
    
    # MPS 디바이스 테스트
    try:
        device = torch.device("mps")
        x = torch.ones(1, device=device)
        y = x * 2
        print(f"✅ MPS 연산 테스트 성공")
        print(f"   텐서: {y} (디바이스: {y.device})")
    except Exception as e:
        print(f"❌ MPS 연산 테스트 실패: {e}")
else:
    mps_available = False
    print("❌ MPS 가속화 지원되지 않음")
    if platform.system() == "Darwin" and "arm" in platform.processor().lower():
        print("   Apple Silicon Mac에서 PyTorch 설치를 확인하세요.")
    else:
        print("   MPS는 Apple Silicon Mac에서만 지원됩니다.")

# CUDA 지원 확인 (Mac에서는 지원 안됨)
print_separator()
print("🔍 CUDA 지원 확인:")
if torch.cuda.is_available():
    print(f"✅ CUDA 지원됨 (비정상 - Mac에서는 일반적으로 지원되지 않음)")
    print(f"   CUDA 버전: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name()}")
else:
    print("ℹ️  CUDA 지원되지 않음 (Mac에서는 정상)")

# CPU 테스트
print_separator()
print("🔍 CPU 테스트:")
try:
    x_cpu = torch.ones(1, device="cpu")
    y_cpu = x_cpu * 2
    print(f"✅ CPU 연산 테스트 성공: {y_cpu}")
except Exception as e:
    print(f"❌ CPU 연산 테스트 실패: {e}")

# 권장 설정
print_separator()
print("💡 권장 설정:")
if mps_available:
    print("""
    # PyTorch 코드에서 다음과 같이 MPS를 사용하세요:
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    inputs = inputs.to(device)
    """)
else:
    print("""
    # PyTorch 코드에서 다음과 같이 CPU를 사용하세요:
    
    device = torch.device('cpu')
    model = model.to(device)
    inputs = inputs.to(device)
    """)

print_separator()
print("테스트 완료!")
print_separator() 