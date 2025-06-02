# Mac 환경 설치 가이드 - CalmMan 얼굴 분류 프로젝트

## 1. 시스템 요구사항
- macOS 10.15 (Catalina) 이상
- Python 3.8 ~ 3.11 권장 (3.12는 일부 패키지와 호환성 이슈 가능)
- 최소 4GB 이상의 사용 가능한 저장 공간

## 2. 가상환경 설정

### 2.1 venv 사용 (기본 방법)
```bash
# 터미널 열기
# 프로젝트 디렉토리로 이동
cd 프로젝트_경로

# 가상환경 생성
python3 -m venv calm-env

# 가상환경 활성화
source calm-env/bin/activate

# pip 업그레이드
pip install --upgrade pip
```

### 2.2 conda 사용 (대안)
```bash
# Miniconda가 설치되어 있지 않다면, 설치 필요: https://docs.conda.io/en/latest/miniconda.html

# 환경 생성
conda create -n calmman-face python=3.10 -y

# 환경 활성화
conda activate calmman-face

# conda-forge 채널 추가
conda config --add channels conda-forge
```

## 3. 의존성 설치

### 3.1 Mac (Apple Silicon - M1/M2/M3)
```bash
# PyTorch MPS 가속 지원 버전 설치
pip install torch torchvision torchaudio

# 기본 의존성 설치
pip install timm opencv-python-headless Pillow scikit-learn matplotlib seaborn tqdm PyYAML

# 의존성 충돌 해결
pip install "numpy<2.0"
pip install albumentations

# facenet-pytorch 의존성 무시 설치 (torch 버전 충돌 해결)
pip install --no-deps facenet-pytorch

# 추가 필요 패키지
pip install pandas pytubefix
```

### 3.2 Mac (Intel)
```bash
# PyTorch CPU 버전 설치
pip install torch torchvision torchaudio

# 나머지 의존성은 동일하게 설치
pip install timm opencv-python-headless Pillow scikit-learn matplotlib seaborn tqdm PyYAML
pip install "numpy<2.0"
pip install albumentations
pip install --no-deps facenet-pytorch
pip install pandas pytubefix
```

## 4. 설치 확인

### 4.1 PyTorch MPS 확인 (Apple Silicon용)
```python
# test_mps.py 파일 생성 후 실행

import torch

print(f"PyTorch 버전: {torch.__version__}")
print(f"MPS 사용 가능: {torch.backends.mps.is_available()}")
print(f"MPS 현재 활성화: {torch.backends.mps.is_built()}")

# MPS 디바이스 테스트
if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(1, device=device)
    print(f"MPS 테스트 성공: {x}")
    print(f"디바이스: {x.device}")
else:
    print("MPS를 사용할 수 없습니다")
```

### 4.2 전체 패키지 확인
```bash
# 터미널에서 실행
python -c "import torch, torchvision, timm, cv2, PIL, facenet_pytorch, numpy, sklearn, albumentations, matplotlib, seaborn, tqdm, yaml; print('모든 패키지가 정상적으로 설치되었습니다!')"
```

## 5. 주의사항 및 팁

### 5.1 성능 최적화
- **Apple Silicon Mac**: PyTorch는 MPS 프레임워크를 통해 GPU 가속화를 지원합니다. 코드에서 `device = torch.device("mps")` 설정으로 활용 가능합니다.
- **Intel Mac**: CPU 모드로 동작하므로 처리 속도가 느릴 수 있습니다.

### 5.2 일반적인 문제 해결
1. **OpenCV 관련 오류**: Mac에서는 `opencv-python-headless` 사용을 권장합니다.
2. **numpy 버전 충돌**: PyTorch는 NumPy 2.0 미만 버전을 권장하므로 `numpy<2.0` 설치를 권장합니다.
3. **facenet-pytorch 의존성 오류**: `--no-deps` 옵션으로 의존성 검사를 건너뛰어 설치합니다.

### 5.3 가상환경 비활성화
```bash
# venv 비활성화
deactivate

# conda 비활성화
conda deactivate
```

## 6. 프로젝트 실행

가상환경을 활성화한 상태에서 프로젝트 스크립트를 실행하세요:

```bash
# 가상환경 활성화
source calm-env/bin/activate  # venv 사용 시
# 또는
conda activate calmman-face  # conda 사용 시

# 설치 테스트 스크립트 실행
python test_installation.py

# 모델 훈련 실행 (예시)
python src/torch_eff_classifier.py
``` 