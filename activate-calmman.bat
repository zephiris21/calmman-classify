@echo off
echo 🚀 침착맨 프로젝트 환경 활성화 중...

REM conda 초기화 (필요시)
call %USERPROFILE%\anaconda3\Scripts\activate.bat

REM 환경 활성화
call conda activate calmman-gpu

REM 환경 확인
echo.
echo ✅ 환경 활성화 완료!
echo 현재 환경: %CONDA_DEFAULT_ENV%
echo.

REM Python 및 CUDA 버전 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

echo.
echo 💡 프로젝트 디렉토리로 이동하려면:
echo    cd /d D:\my_projects\calmman-facial-classification
echo.

REM 명령 프롬프트 유지
cmd /k