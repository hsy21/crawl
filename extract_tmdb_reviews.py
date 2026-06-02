import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import csv
import json
import os
import time

# 요청 재시도를 위한 Session 설정
session = requests.Session()
retry = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
adapter = HTTPAdapter(max_retries=retry)
session.mount("http://", adapter)
session.mount("https://", adapter)

API_KEY = "823b0e178b09a8d4a6862c5f6b5064c9"
ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI4MjNiMGUxNzhiMDlhOGQ0YTY4NjJjNWY2YjUwNjRjOSIsIm5iZiI6MTc3OTE2NDA5MC42MzY5OTk4LCJzdWIiOiI2YTBiZTNiYWE3Y2IwMTljZTVjMjc3Y2UiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.VBKYeSdy8JbwrBqVZ6eRfr0BRvexaqv2dS-gdKL620g"

HEADERS = {
    "accept": "application/json",
    "Authorization": f"Bearer {ACCESS_TOKEN}"
}

# 스크립트 실행 위치와 무관하게 항상 _AIService26 폴더에 데이터가 저장되도록 절대 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_FILE = os.path.join(BASE_DIR, "movie_dataset.csv")
STATE_FILE = os.path.join(BASE_DIR, "tmdb_progress.json")

# 이미 저장된 리뷰와 영화를 추적하여 중복 저장을 방지하고 속도를 높이기 위한 집합(Set)
seen_reviews = set()
seen_movies = set()

def init_seen_reviews():
    """CSV 파일을 읽어서 이미 저장된 영화 ID와 리뷰 내용을 집합에 저장합니다."""
    if os.path.exists(CSV_FILE):
        print("기존 CSV 파일에서 중복 검사를 위한 데이터를 불러오는 중...")
        with open(CSV_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "content" in reader.fieldnames:
                for row in reader:
                    movie_id = row.get("movie_id", "").strip()
                    content = row.get("content", "").strip()
                    
                    if movie_id:
                        seen_movies.add(movie_id)
                    if content:
                        seen_reviews.add(content)
        print(f"총 {len(seen_movies)}개의 이미 수집된 영화와 {len(seen_reviews)}개의 기존 리뷰를 로드했습니다.")

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            # 이전 버전 또는 4월 이후에서 시작하려는 경우 2026년 4월로 리셋
            if "month" not in state or (state.get("year") == 2026 and state.get("month") > 4):
                print("진행 상태가 2026년 4월 이후를 가리키거나 옛날 버전입니다. 2026년 4월부터 시작하도록 초기화합니다.")
                return {"year": 2026, "month": 4, "page": 1, "movie_index": 0}
            return state
    return {"year": 2026, "month": 4, "page": 1, "movie_index": 0}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

def get_days_in_month(year, month):
    if month in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    elif month in [4, 6, 9, 11]:
        return 30
    elif month == 2:
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            return 29
        return 28
    return 31

def append_to_csv(data_list):
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["movie_id", "title", "rating", "content"])
        writer.writerows(data_list)

def get_movies(year, month, page):
    # API 한계를 극복하기 위해 연도+월별로 쪼개어 100만건 이상의 방대한 데이터를 수집할 수 있도록 변경
    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year}-{month:02d}-{get_days_in_month(year, month)}"
        
    # 이미 유명한 영화는 다 수집했으므로, 더 많은 리뷰를 찾기 위해 기준을 대폭 낮춤 (vote_count.gte=5)
    url = f"https://api.themoviedb.org/3/discover/movie?include_adult=false&include_video=false&language=en-US&page={page}&primary_release_date.gte={start_date}&primary_release_date.lte={end_date}&sort_by=vote_count.desc&vote_count.gte=5"
    for attempt in range(3):
        try:
            response = session.get(url, headers=HEADERS, timeout=15)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                time.sleep(2)
        except Exception as e:
            print(f"  [오류] 영화 목록 가져오기 실패: {e}")
            time.sleep(3)
    return None

def get_reviews(movie_id):
    # language=en-US 파라미터를 제거하여 영어뿐만 아니라 모든 언어의 리뷰를 최대한 수집하도록 변경
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?page=1"
    reviews = []
    retry_count = 0
    while url:
        if retry_count > 3:
            print(f"  [오류] 최대 재시도 초과. 리뷰 수집 중단 (movie_id: {movie_id})")
            break
        try:
            response = session.get(url, headers=HEADERS, timeout=15)
            if response.status_code == 200:
                data = response.json()
                reviews.extend(data.get("results", []))
                if data.get("page", 1) < data.get("total_pages", 1):
                    url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?page={data.get('page') + 1}"
                    retry_count = 0 # 다음 페이지로 넘어갔으므로 초기화
                else:
                    url = None
            elif response.status_code == 429: # Too Many Requests
                print("  [경고] API 요청 한도 초과. 잠시 대기합니다...")
                time.sleep(2)
                retry_count += 1
                continue
            else:
                print(f"  [경고] 리뷰 가져오기 실패 (상태 코드: {response.status_code})")
                url = None
        except requests.exceptions.JSONDecodeError:
            print(f"  [오류] JSON 파싱 오류 발생 (movie_id: {movie_id})")
            url = None
        except Exception as e:
            print(f"  [오류] 리뷰 가져오기 실패: {e}")
            time.sleep(3)
            retry_count += 1
    return reviews

def main():
    # 기존 리뷰 목록 로드 (중복 제거용)
    init_seen_reviews()
    
    state = load_state()
    current_year = state["year"]
    current_month = state.get("month", 12)
    current_page = state["page"]
    current_movie_index = state["movie_index"]

    print("TMDB 데이터 수집을 시작합니다. 언제든지 Ctrl+C를 눌러 중지할 수 있습니다.")
    print(f"현재 진행 위치 - 연도: {current_year}, 월: {current_month}, 페이지: {current_page}, 영화 인덱스: {current_movie_index}")

    try:
        # 2010년까지 100만건 이상의 수집을 위해 2009년 이전(2010년 포함)까지 탐색
        for year in range(current_year, 2009, -1):
            if year == 2026:
                # 2026년은 최대 4월부터 시작
                start_month = current_month if current_month <= 4 else 4
            else:
                start_month = current_month if year == current_year else 12
                
            for month in range(start_month, 0, -1):
                while True:
                    print(f"[{year}-{month:02d}] 페이지 {current_page} 영화 목록 가져오는 중...")
                    movies_data = get_movies(year, month, current_page)
                    if not movies_data:
                        break
                    
                    movies = movies_data.get("results", [])
                    total_pages = movies_data.get("total_pages", 1)
                    
                    if not movies:
                        break
                    
                    for idx in range(current_movie_index, len(movies)):
                        movie = movies[idx]
                        movie_id = str(movie["id"])
                        title = movie.get("title", "")
                        
                        # 이미 수집한 이력이 있는 영화라면 API 호출을 생략하여 속도를 대폭 높임
                        if movie_id in seen_movies:
                            print(f"  [건너뜀] 이미 리뷰를 수집한 영화: '{title}'")
                            continue
                            
                        reviews = get_reviews(movie_id)
                        
                        # 수집 시도한 영화는 리뷰가 없더라도 다음번엔 건너뛰도록 처리
                        seen_movies.add(movie_id)
                        
                        csv_data = []
                        for review in reviews:
                            rating = review.get("author_details", {}).get("rating", "")
                            content = review.get("content", "").replace('\n', ' ').replace('\r', '').strip()
                            
                            # 중복된 리뷰 내용이거나 내용이 비어있으면 건너뛰기
                            if not content or content in seen_reviews:
                                continue
                                
                            seen_reviews.add(content)
                            csv_data.append([movie_id, title, rating, content])
                        
                        if csv_data:
                            append_to_csv(csv_data)
                            print(f"  [수집 완료] '{title}'에서 {len(csv_data)}개의 새로운 리뷰가 수집되었습니다.")
                        
                        # 리뷰 추출 후 상태 저장
                        state["year"] = year
                        state["month"] = month
                        state["page"] = current_page
                        state["movie_index"] = idx + 1
                        save_state(state)
                    
                    current_page += 1
                    current_movie_index = 0
                    state["page"] = current_page
                    state["movie_index"] = 0
                    save_state(state)
                    
                    # TMDB Max=500p
                    if current_page > total_pages or current_page > 500:
                        break
                
                # 다음 월로 넘어가면 페이지 초기화
                current_page = 1
                current_movie_index = 0
                state["month"] = month - 1 if month > 1 else 12
                state["page"] = current_page
                state["movie_index"] = current_movie_index
                save_state(state)
            
            # 다음 연도로 넘어가기 위한 초기화
            current_month = 12
            current_page = 1
            current_movie_index = 0
            state["year"] = year - 1
            state["month"] = current_month
            state["page"] = current_page
            state["movie_index"] = current_movie_index
            save_state(state)
            
        print("모든 수집이 완료되었습니다!")
            
    except KeyboardInterrupt:
        print("\n사용자에 의해 프로세스가 중지되었습니다. 진행 상황이 저장되었습니다.")
        print(f"다음 실행 시 연도: {state['year']}, 월: {state.get('month', 12)}, 페이지: {state['page']}, 영화 인덱스: {state['movie_index']} 부터 재개됩니다.")

if __name__ == "__main__":
    main()
