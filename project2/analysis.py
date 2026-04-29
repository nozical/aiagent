import time
import os
import glob
from google import genai

# 1. API 키 설정 - GitHub Secrets의 GOOGLE_API_KEY 환경변수에서 가져옴
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")

client = genai.Client(api_key=GOOGLE_API_KEY)

# 2. 동영상 파일 경로 설정 - project2/data/ 폴더에서 영상 파일 자동 탐색
# 지원 확장자: mp4, mov, avi, mkv, webm
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# data 폴더에서 영상 파일 탐색
VIDEO_EXTENSIONS = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm"]
video_files = []
for ext in VIDEO_EXTENSIONS:
    video_files.extend(glob.glob(os.path.join(DATA_DIR, ext)))

if not video_files:
    raise FileNotFoundError(
        f"'{DATA_DIR}' 폴더에 영상 파일이 없습니다. "
        "mp4, mov, avi, mkv, webm 형식의 파일을 넣어주세요."
    )

print(f"분석할 영상 파일 {len(video_files)}개 발견:")
for f in video_files:
    print(f"  - {os.path.basename(f)}")

# 3. 모델 설정
MODEL_ID = "gemini-2.5-flash"

# 분석 프롬프트
PROMPT = "이 동영상에서 인물이 어떤 행동을 하고 있는지 시간 흐름에 따라 상세히 분석해줘."

# 전체 결과 저장용
all_results = []

# 4. 영상 파일별 분석
for video_path in video_files:
    video_name = os.path.basename(video_path)
    print(f"\n{'='*50}")
    print(f"처리 중: {video_name}")
    print(f"{'='*50}")

    video_file = None
    try:
        # 파일 업로드
        print(f"업로드 중...")
        video_file = client.files.upload(file=video_path)
        print(f"업로드 완료: {video_file.uri}")

        # 처리 대기
        print("Google 서버에서 처리 중...")
        while video_file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(5)
            video_file = client.files.get(name=video_file.name)
        print()

        if video_file.state.name == "FAILED":
            raise ValueError(f"동영상 처리 실패: {video_name}")

        print("처리 완료! AI 분석 시작...")

        # AI 분석 요청
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[video_file, PROMPT]
        )

        result_text = response.text
        print(f"\n[분석 결과 - {video_name}]")
        print(result_text)

        all_results.append({
            "file": video_name,
            "result": result_text
        })

    finally:
        # 업로드된 파일 서버에서 삭제
        if video_file:
            try:
                client.files.delete(name=video_file.name)
                print(f"\n서버에서 '{video_name}' 파일 삭제 완료.")
            except Exception as e:
                print(f"\n파일 삭제 중 오류: {e}")

# 5. 결과를 텍스트 파일로 저장
result_file_path = os.path.join(RESULT_DIR, "analysis_result.txt")
with open(result_file_path, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("  Gemini AI 영상 분석 결과 보고서\n")
    f.write("=" * 60 + "\n\n")
    for item in all_results:
        f.write(f"【파일명】 {item['file']}\n")
        f.write("-" * 40 + "\n")
        f.write(item["result"])
        f.write("\n\n")

print(f"\n결과 파일 저장 완료: {result_file_path}")
