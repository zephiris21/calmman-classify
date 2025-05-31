#!/usr/bin/env python3
"""
GPU 및 필수 라이브러리 테스트 스크립트
"""

def test_pytorch():
    try:
        import torch
        print("=" * 50)
        print("PyTorch 테스트")
        print("=" * 50)
        print(f"PyTorch 버전: {torch.__version__}")
        print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA 버전: {torch.version.cuda}")
            print(f"GPU 개수: {torch.cuda.device_count()}")
            print(f"현재 GPU: {torch.cuda.current_device()}")
            print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
            
            # 간단한 텐서 연산 테스트
            x = torch.randn(3, 3).cuda()
            y = torch.randn(3, 3).cuda()
            z = x @ y
            print(f"GPU 연산 테스트: 성공 (결과 크기: {z.shape})")
        else:
            print("❌ CUDA를 사용할 수 없습니다.")
            
    except ImportError:
        print("❌ PyTorch가 설치되지 않았습니다.")
    except Exception as e:
        print(f"❌ PyTorch 오류: {e}")

def test_facenet():
    try:
        from facenet_pytorch import MTCNN
        import torch
        
        print("\n" + "=" * 50)
        print("MTCNN 테스트")
        print("=" * 50)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mtcnn = MTCNN(device=device)
        print(f"MTCNN 초기화 성공 (device: {device})")
        
    except ImportError:
        print("\n❌ facenet-pytorch가 설치되지 않았습니다.")
    except Exception as e:
        print(f"\n❌ MTCNN 오류: {e}")

def test_other_libraries():
    print("\n" + "=" * 50)
    print("기타 라이브러리 테스트")
    print("=" * 50)
    
    libraries = [
        'cv2', 'numpy', 'PIL', 'matplotlib', 
        'sklearn', 'albumentations', 'tqdm'
    ]
    
    for lib in libraries:
        try:
            __import__(lib)
            print(f"✅ {lib}: 설치됨")
        except ImportError:
            print(f"❌ {lib}: 설치되지 않음")

def test_pytubefix():
    try:
        from pytubefix import YouTube
        print("\n" + "=" * 50)
        print("PyTubeFix 테스트")
        print("=" * 50)
        print("✅ pytubefix: 설치됨")
        
    except ImportError:
        print("\n❌ pytubefix가 설치되지 않았습니다.")

def test_tensorflow():
    try:
        import tensorflow as tf
        print("\n" + "=" * 50)
        print("TensorFlow 테스트")
        print("=" * 50)
        print(f"TensorFlow 버전: {tf.__version__}")
        
        # GPU 디바이스 확인
        gpu_devices = tf.config.list_physical_devices('GPU')
        print(f"GPU 디바이스 개수: {len(gpu_devices)}")
        
        if len(gpu_devices) > 0:
            print("⚠️ TensorFlow가 GPU를 감지했습니다. CPU 강제 사용을 권장합니다.")
            # GPU 숨기기 (선택사항)
            # tf.config.set_visible_devices([], 'GPU')
        else:
            print("✅ TensorFlow CPU 모드로 실행됩니다.")
        
        # 간단한 연산 테스트
        x = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
        y = tf.constant([[2, 0], [0, 2]], dtype=tf.float32)
        z = tf.matmul(x, y)
        print(f"CPU 연산 테스트: 성공 (결과 크기: {z.shape})")
        
    except ImportError:
        print("\n❌ TensorFlow가 설치되지 않았습니다.")
    except Exception as e:
        print(f"\n❌ TensorFlow 오류: {e}")

if __name__ == "__main__":
    print("🚀 침착맨 프로젝트 환경 테스트 시작")
    
    test_pytorch()
    test_facenet()
    test_other_libraries()
    test_pytubefix()
    test_tensorflow()  
    
    print("\n" + "=" * 50)
    print("테스트 완료!")
    print("=" * 50)