#!/bin/bash

# 폴더 경로 가져오기
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 색상 정의
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}=== CalmMan 얼굴 분류 프로젝트 환경 활성화 ===${NC}"
echo -e "${CYAN}스크립트 위치: ${SCRIPT_DIR}${NC}"

# 가상환경 경로
VENV_PATH="${SCRIPT_DIR}/calm-env"

# 가상환경 존재 여부 확인
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${YELLOW}가상환경이 발견되지 않았습니다. 새로 생성합니다...${NC}"
    
    # Python 버전 확인
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo -e "${CYAN}Python 버전: ${PYTHON_VERSION}${NC}"
    
    # 가상환경 생성
    python3 -m venv "$VENV_PATH"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}가상환경 생성에 실패했습니다.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}가상환경이 성공적으로 생성되었습니다.${NC}"
    
    # 가상환경 활성화
    source "${VENV_PATH}/bin/activate"
    
    # pip 업그레이드
    echo -e "${CYAN}pip를 최신 버전으로 업그레이드합니다...${NC}"
    pip install --upgrade pip
    
    # 의존성 설치 안내
    echo -e "${YELLOW}의존성 설치가 필요합니다. 다음 명령어를 실행하세요:${NC}"
    echo -e "${CYAN}Mac (Apple Silicon - M1/M2/M3) 사용자:${NC}"
    echo -e "1. pip install torch torchvision torchaudio"
    echo -e "2. pip install timm opencv-python-headless Pillow scikit-learn matplotlib seaborn tqdm PyYAML"
    echo -e "3. pip install \"numpy<2.0\" albumentations"
    echo -e "4. pip install --no-deps facenet-pytorch"
    echo -e "5. pip install pandas pytubefix"
    echo -e "${CYAN}Mac (Intel) 사용자:${NC}"
    echo -e "동일한 명령어를 실행하세요."
    
    # MPS 테스트 안내
    echo -e "${YELLOW}설치 후 'python test_mps.py'를 실행하여 MPS 지원을 확인하세요.${NC}"
else
    # 가상환경 활성화
    source "${VENV_PATH}/bin/activate"
    echo -e "${GREEN}가상환경이 활성화되었습니다.${NC}"
    
    # 현재 설치된 패키지 정보 출력
    echo -e "${CYAN}설치된 주요 패키지:${NC}"
    pip list | grep -E "torch|torchvision|opencv|numpy|albumentations|facenet|scikit|matplotlib"
    
    # 실행 안내
    echo -e "${GREEN}환경이 준비되었습니다. 이제 프로젝트 스크립트를 실행할 수 있습니다.${NC}"
fi

# 시스템 정보 출력
echo -e "${CYAN}시스템 정보:${NC}"
echo -e "OS: $(uname -s) $(uname -r)"
echo -e "프로세서: $(uname -p)"

# Apple Silicon 감지
if [[ "$(uname -p)" == "arm"* ]]; then
    echo -e "${GREEN}Apple Silicon이 감지되었습니다. MPS 가속화를 사용할 수 있습니다.${NC}"
else
    echo -e "${YELLOW}Intel Mac이 감지되었습니다. CPU 모드로 실행됩니다.${NC}"
fi

# 현재 작업 디렉토리 유지
cd "$SCRIPT_DIR"

echo -e "${CYAN}=======================================================${NC}"
echo -e "${GREEN}가상환경이 성공적으로 활성화되었습니다: calm-env${NC}"
echo -e "${YELLOW}비활성화하려면 'deactivate' 명령어를 입력하세요.${NC}"
echo -e "${CYAN}=======================================================${NC}" 