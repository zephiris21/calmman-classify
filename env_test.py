#!/usr/bin/env python3
"""
침착맨 프로젝트 - GPU 및 필수 라이브러리 완전 테스트 스크립트
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
            print("⚠️ CUDA를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
            # CPU 연산 테스트
            x = torch.randn(3, 3)
            y = torch.randn(3, 3)
            z = x @ y
            print(f"CPU 연산 테스트: 성공 (결과 크기: {z.shape})")
            
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
        print(f"✅ MTCNN 초기화 성공 (device: {device})")
        
    except ImportError:
        print("\n❌ facenet-pytorch가 설치되지 않았습니다.")
    except Exception as e:
        print(f"\n❌ MTCNN 오류: {e}")

def test_facenet_detailed():
    try:
        from facenet_pytorch import InceptionResnetV1
        import torch
        
        print("\n" + "=" * 50)
        print("FaceNet InceptionResnetV1 테스트")
        print("=" * 50)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # VGGFace2 사전훈련 모델 로드 테스트
        try:
            facenet = InceptionResnetV1(pretrained='vggface2')
            facenet.eval()
            print(f"✅ VGGFace2 사전훈련 모델 로드 성공")
            print(f"   디바이스: {device}")
            
            # 더미 입력으로 추론 테스트
            dummy_input = torch.randn(1, 3, 160, 160)
            with torch.no_grad():
                embedding = facenet(dummy_input)
            print(f"✅ 임베딩 추출 테스트 성공 (크기: {embedding.shape})")
            
        except Exception as e:
            print(f"⚠️ VGGFace2 모델 다운로드 실패 (인터넷 연결 확인): {e}")
        
    except ImportError:
        print("\n❌ facenet-pytorch InceptionResnetV1이 설치되지 않았습니다.")
    except Exception as e:
        print(f"\n❌ FaceNet 상세 테스트 오류: {e}")

def test_timm():
    try:
        import timm
        import torch
        print("\n" + "=" * 50)
        print("Timm (EfficientNet) 테스트")
        print("=" * 50)
        print(f"Timm 버전: {timm.__version__}")
        
        # EfficientNet-B0 모델 로드 테스트
        model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
        print(f"✅ EfficientNet-B0 모델 생성 성공")
        print(f"   파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
        
        # 사전훈련 모델 다운로드 테스트 (선택적)
        try:
            model_pretrained = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
            print(f"✅ ImageNet 사전훈련 모델 다운로드 성공")
            
            # 간단한 추론 테스트
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                features = model_pretrained(dummy_input)
            print(f"✅ 특징 추출 테스트 성공 (크기: {features.shape})")
            
        except Exception as e:
            print(f"⚠️ 사전훈련 모델 다운로드 실패 (인터넷 연결 확인): {e}")
        
    except ImportError:
        print("\n❌ timm이 설치되지 않았습니다.")
    except Exception as e:
        print(f"\n❌ timm 오류: {e}")

def test_other_libraries():
    print("\n" + "=" * 50)
    print("기타 라이브러리 테스트")
    print("=" * 50)
    
    libraries = {
        'cv2': 'OpenCV',
        'numpy': 'NumPy', 
        'PIL': 'Pillow',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'sklearn': 'Scikit-learn',
        'albumentations': 'Albumentations',
        'tqdm': 'TQDM',
        'torchvision': 'TorchVision'
    }
    
    for module, name in libraries.items():
        try:
            lib = __import__(module)
            # 버전 정보가 있으면 출력
            version = getattr(lib, '__version__', '버전 정보 없음')
            print(f"✅ {name}: {version}")
        except ImportError:
            print(f"❌ {name}: 설치되지 않음")

def test_yaml():
    try:
        import yaml
        import os
        print("\n" + "=" * 50)
        print("YAML 설정 파일 테스트")
        print("=" * 50)
        print(f"✅ PyYAML 버전: {yaml.__version__}")
        
        # 설정 파일 존재 여부 확인
        config_files = [
            'config/config_torch.yaml',
            'config/config.yaml'
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    print(f"✅ {config_file}: 로드 성공")
                    
                    # 주요 설정 확인
                    if 'classifier' in config:
                        device = config['classifier'].get('device', '설정 없음')
                        print(f"   분류기 디바이스 설정: {device}")
                    if 'face_recognition' in config:
                        enabled = config['face_recognition'].get('enabled', False)
                        print(f"   얼굴 인식 활성화: {enabled}")
                        
                except Exception as e:
                    print(f"❌ {config_file}: 로드 실패 - {e}")
            else:
                print(f"⚠️ {config_file}: 파일이 존재하지 않음")
        
    except ImportError:
        print("\n❌ PyYAML이 설치되지 않았습니다.")
    except Exception as e:
        print(f"\n❌ YAML 오류: {e}")

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

def test_system_info():
    try:
        import platform
        import psutil
        
        print("\n" + "=" * 50)
        print("시스템 정보")
        print("=" * 50)
        
        print(f"OS: {platform.system()} {platform.release()}")
        print(f"Python 버전: {platform.python_version()}")
        print(f"아키텍처: {platform.machine()}")
        print(f"프로세서: {platform.processor()}")
        print(f"CPU 코어 수: {psutil.cpu_count(logical=False)} (물리) / {psutil.cpu_count(logical=True)} (논리)")
        
        memory = psutil.virtual_memory()
        print(f"총 메모리: {memory.total / (1024**3):.1f} GB")
        print(f"사용 가능 메모리: {memory.available / (1024**3):.1f} GB")
        print(f"메모리 사용률: {memory.percent:.1f}%")
        
        # GPU 메모리 정보 (PyTorch 기준)
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_props = torch.cuda.get_device_properties(i)
                    gpu_memory = gpu_props.total_memory / (1024**3)
                    print(f"GPU {i} 메모리: {gpu_memory:.1f} GB ({gpu_props.name})")
        except:
            pass
            
    except ImportError:
        print("\n❌ psutil이 설치되지 않았습니다.")
        print("기본 시스템 정보만 표시합니다.")
        import platform
        print(f"OS: {platform.system()} {platform.release()}")
        print(f"Python 버전: {platform.python_version()}")
    except Exception as e:
        print(f"\n❌ 시스템 정보 오류: {e}")

def test_inference_pipeline():
    try:
        import torch
        import numpy as np
        from PIL import Image
        
        print("\n" + "=" * 50)
        print("추론 파이프라인 테스트")
        print("=" * 50)
        
        # 더미 이미지 생성 (224x224x3)
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_image = Image.fromarray(dummy_image)
        print(f"✅ 더미 이미지 생성 성공 (크기: {dummy_image.shape})")
        
        # PyTorch transform 테스트
        try:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            tensor_image = transform(pil_image)
            print(f"✅ 이미지 전처리 성공 (크기: {tensor_image.shape})")
        except Exception as e:
            print(f"❌ 이미지 전처리 실패: {e}")
        
        # MTCNN 얼굴 탐지 테스트
        try:
            from facenet_pytorch import MTCNN
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            mtcnn = MTCNN(device=device, image_size=224, margin=20)
            
            # 더미 이미지에서 얼굴 탐지 (실패해도 정상)
            faces = mtcnn(pil_image)
            if faces is not None:
                print(f"✅ MTCNN 얼굴 탐지 성공 (크기: {faces.shape})")
            else:
                print(f"⚠️ MTCNN 얼굴 미탐지 (더미 이미지이므로 정상)")
        except Exception as e:
            print(f"❌ MTCNN 테스트 실패: {e}")
        
        # OpenCV 이미지 처리 테스트
        try:
            import cv2
            # 더미 이미지 처리
            gray = cv2.cvtColor(dummy_image, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(dummy_image, (112, 112))
            print(f"✅ OpenCV 이미지 처리 성공 (그레이: {gray.shape}, 리사이즈: {resized.shape})")
        except Exception as e:
            print(f"❌ OpenCV 테스트 실패: {e}")
        
    except Exception as e:
        print(f"\n❌ 추론 파이프라인 테스트 오류: {e}")

def test_directory_structure():
    import os
    
    print("\n" + "=" * 50)
    print("프로젝트 디렉토리 구조 확인")
    print("=" * 50)
    
    expected_dirs = [
        'config',
        'src',
        'data',
        'results',
        'models'
    ]
    
    expected_files = [
        'src/torch_video_processor.py',
        'src/pytorch_classifier.py',
        'src/torch_eff_classifier.py',
        'config/config_torch.yaml'
    ]
    
    print("📁 예상 디렉토리:")
    for dir_name in expected_dirs:
        if os.path.exists(dir_name):
            file_count = len([f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))])
            print(f"✅ {dir_name}/ ({file_count}개 파일)")
        else:
            print(f"❌ {dir_name}/ (없음)")
    
    print("\n📄 핵심 파일:")
    for file_name in expected_files:
        if os.path.exists(file_name):
            size = os.path.getsize(file_name) / 1024  # KB
            print(f"✅ {file_name} ({size:.1f} KB)")
        else:
            print(f"❌ {file_name} (없음)")
    
    # 현재 디렉토리의 실제 구조 표시
    print(f"\n📋 현재 디렉토리 구조:")
    current_dir = os.getcwd()
    print(f"작업 디렉토리: {current_dir}")
    
    for item in sorted(os.listdir('.')):
        if os.path.isdir(item):
            print(f"📁 {item}/")
        else:
            print(f"📄 {item}")

def test_model_files():
    import os
    
    print("\n" + "=" * 50)
    print("모델 파일 확인")
    print("=" * 50)
    
    model_dirs = [
        'models',
        'results/pytorch_efficientnet/models',
        'src/models'
    ]
    
    model_extensions = ['.pth', '.pt', '.pkl', '.h5']
    
    found_models = []
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            print(f"📁 {model_dir}/:")
            for file in os.listdir(model_dir):
                file_path = os.path.join(model_dir, file)
                if os.path.isfile(file_path):
                    _, ext = os.path.splitext(file)
                    if ext.lower() in model_extensions:
                        size = os.path.getsize(file_path) / (1024*1024)  # MB
                        print(f"  ✅ {file} ({size:.1f} MB)")
                        found_models.append(file_path)
                    else:
                        print(f"  📄 {file}")
        else:
            print(f"❌ {model_dir}/ (없음)")
    
    if found_models:
        print(f"\n🎯 총 {len(found_models)}개의 모델 파일을 찾았습니다.")
    else:
        print(f"\n⚠️ 모델 파일을 찾을 수 없습니다. 먼저 모델을 훈련해주세요.")

def test_data_structure():
    import os
    
    print("\n" + "=" * 50)
    print("데이터 디렉토리 구조 확인")
    print("=" * 50)
    
    data_dirs = [
        'data/processed/teasing',
        'data/processed/non_teasing',
        'data/raw',
        'data/videos'
    ]
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            video_files = [f for f in files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            
            print(f"✅ {data_dir}/:")
            if image_files:
                print(f"  🖼️ 이미지: {len(image_files)}개")
            if video_files:
                print(f"  🎬 비디오: {len(video_files)}개")
            if not files:
                print(f"  📭 빈 폴더")
        else:
            print(f"❌ {data_dir}/ (없음)")

def print_summary():
    print("\n" + "🎯" * 50)
    print("테스트 요약 및 권장사항")
    print("🎯" * 50)
    
    print("\n✅ 필수 패키지 (모두 ✅ 표시되어야 함):")
    print("   - PyTorch (torch, torchvision)")
    print("   - Timm (EfficientNet)")
    print("   - FaceNet-PyTorch (MTCNN, InceptionResnetV1)")
    print("   - OpenCV (cv2)")
    print("   - NumPy, PIL, Scikit-learn")
    print("   - PyYAML")
    
    print("\n⚠️ 경고가 발생할 수 있는 항목:")
    print("   - VGGFace2 모델 다운로드 (인터넷 연결 필요)")
    print("   - ImageNet 사전훈련 모델 다운로드 (인터넷 연결 필요)")
    print("   - 설정 파일 또는 모델 파일 누락")
    
    print("\n💡 성능 최적화 팁:")
    print("   - GPU 사용 가능하면 자동으로 활용됩니다")
    print("   - CPU만 사용하는 경우 배치 크기를 줄이세요")
    print("   - 메모리 부족 시 frame_skip 값을 증가시키세요")
    
    print("\n📝 다음 단계:")
    print("   1. 모든 ✅ 확인되면 프로젝트 실행 가능")
    print("   2. ❌ 항목이 있으면 해당 패키지 설치")
    print("   3. 설정 파일과 데이터 구조 확인")
    print("   4. 모델 훈련 또는 사전훈련된 모델 로드")

if __name__ == "__main__":
    print("🚀 침착맨 프로젝트 - 완전 환경 테스트 시작")
    print("=" * 60)
    
    # 핵심 라이브러리 테스트
    test_pytorch()
    test_timm()                    # 매우 중요!
    test_facenet()
    test_facenet_detailed()
    test_other_libraries()
    
    # 설정 및 구조 테스트
    test_yaml()
    test_directory_structure()
    test_model_files()
    test_data_structure()
    
    # 시스템 정보
    test_system_info()
    
    # 파이프라인 테스트
    test_inference_pipeline()
    
    # 선택적 라이브러리
    test_pytubefix()
    test_tensorflow()
    
    # 요약
    print_summary()
    
    print("\n" + "🎉" * 50)
    print("완전 환경 테스트 완료!")
    print("🎉" * 50)