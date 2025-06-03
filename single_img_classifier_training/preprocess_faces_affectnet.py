
import os
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append('./src')
from mtcnn_wrapper import FaceDetector

def preprocess_faces_for_affectnet():
    """AffectNet용 얼굴 전처리 (260x260, 정렬)"""
    
    print("=== 👁️ AffectNet용 얼굴 전처리 ===")
    
    # 경로 설정
    input_base = r'D:\my_projects\calmman-facial-classification\data\raw'
    output_base = r'D:\my_projects\calmman-facial-classification\data\affectnet_processed'
    
    # MTCNN 초기화 (260x260, 정렬 활성화)
    detector = FaceDetector(
        image_size=260,
        margin=20,
        prob_threshold=0.9,
        align_faces=True
    )
    
    # 클래스별 처리
    classes = ['teasing', 'non_teasing']
    
    for class_name in classes:
        input_dir = os.path.join(input_base, class_name)
        output_dir = os.path.join(output_base, class_name)
        
        print(f"\n📁 {class_name} 클래스 처리 중...")
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 이미지 파일 수집
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(Path(input_dir).glob(f'*{ext}'))
            image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
        
        print(f"   발견된 이미지: {len(image_files)}개")
        
        success_count = 0
        failed_count = 0
        
        # 각 이미지 처리
        for img_path in tqdm(image_files, desc=f"Processing {class_name}"):
            try:
                img_name = Path(img_path).stem
                output_path = os.path.join(output_dir, f"{img_name}.jpg")
                
                # 이미 처리된 파일은 건너뛰기
                if os.path.exists(output_path):
                    success_count += 1
                    continue
                
                # 얼굴 처리
                success = detector.process_image_to_class_dir(
                    str(img_path), 
                    class_name, 
                    output_base
                )
                
                if success:
                    success_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                print(f"   ⚠️ {img_path} 처리 실패: {e}")
                failed_count += 1
        
        print(f"   ✅ 성공: {success_count}개")
        print(f"   ❌ 실패: {failed_count}개")
        print(f"   📁 저장 위치: {output_dir}")

if __name__ == "__main__":
    preprocess_faces_for_affectnet()