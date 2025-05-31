from pytubefix import YouTube
from pytubefix.cli import on_progress
import os

def download_test_video():
    # 테스트용 침착맨 영상 URL (짧은 영상으로 테스트)
    url = input("침착맨 유튜브 URL을 입력하세요: ")
    
    try:
        # YouTube 객체 생성 (진행상황 콜백 포함)
        yt = YouTube(url, on_progress_callback=on_progress)
        
        # 영상 정보 출력
        print(f"제목: {yt.title}")
        print(f"길이: {yt.length}초")
        print(f"조회수: {yt.views:,}")
        
        # 사용 가능한 화질 옵션 확인
        print("\n사용 가능한 화질:")
        for stream in yt.streams.filter(progressive=True, file_extension='mp4'):
            print(f"  - {stream.resolution} ({stream.filesize_mb:.1f}MB)")
        
        # 720p 또는 최고화질로 다운로드
        stream = (yt.streams.filter(progressive=True, file_extension='mp4', res='720p').first() or
                 yt.streams.filter(progressive=True, file_extension='mp4').get_highest_resolution())
        
        if stream:
            print(f"\n다운로드 시작: {stream.resolution}")
            output_path = os.path.join(os.path.dirname(__file__), "downloads")
            os.makedirs(output_path, exist_ok=True)
            
            stream.download(output_path=output_path)
            print("다운로드 완료!")
        else:
            print("다운로드 가능한 스트림을 찾을 수 없습니다.")
            
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    download_test_video()