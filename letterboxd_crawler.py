import os
import json
import time
import random
from bs4 import BeautifulSoup
from curl_cffi import requests

# ==========================================
# 설정 변수
# ==========================================
START_YEAR = 2016
END_YEAR = 2026
DATASET_FILE = "dataset.json"
MAX_PAGES_PER_MOVIE = 50  # 한 영화당 최대 크롤링할 리뷰 페이지 수 (None이면 무제한)

# 차단 방지를 위한 User-Agent 리스트 및 딜레이 설정
HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
    "Referer": "https://letterboxd.com/",
}

def load_existing_data():
    if os.path.exists(DATASET_FILE):
        try:
            with open(DATASET_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"[{DATASET_FILE}] 기존 데이터 {len(data)}개를 불러왔습니다.")
                return data
        except json.JSONDecodeError:
            print(f"[{DATASET_FILE}] 파일이 손상되었거나 비어있습니다. 새로 시작합니다.")
            return []
    return []

def save_data(data):
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"[{DATASET_FILE}] 데이터가 업데이트 되었습니다. (총 {len(data)}개)")

def random_sleep(min_sec=1.5, max_sec=3.5):
    """봇 방지를 피하기 위해 랜덤하게 대기합니다."""
    time.sleep(random.uniform(min_sec, max_sec))

def get_html(url, max_retries=3):
    """curl_cffi를 사용하여 브라우저(크롬) 환경을 흉내내어 HTML을 가져옵니다."""
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url, headers=HEADERS, impersonate="chrome", timeout=10)
            if response.status_code == 200:
                return response.text
            elif response.status_code == 429:
                wait_time = 10 * (2 ** retries) # Exponential backoff: 10, 20, 40초 대기
                print(f"[경고] Too Many Requests. {wait_time}초 대기합니다... ({url})")
                time.sleep(wait_time)
                retries += 1
            else:
                print(f"[오류] 상태 코드 {response.status_code} - {url}")
                return None
        except Exception as e:
            print(f"[예외 발생] {url} - {str(e)}")
            time.sleep(5)
            retries += 1
            
    print(f"[실패] 최대 재시도 횟수 초과 - {url}")
    return None

def get_movie_slugs_for_year(year):
    """해당 연도에 개봉한 영화들의 슬러그(고유 URL 아이디)를 가져옵니다."""
    slugs = []
    page = 1
    
    while True:
        print(f"[{year}년] 영화 목록 페이지 {page} 탐색 중...")
        url = f"https://letterboxd.com/films/year/{year}/page/{page}/"
        html = get_html(url)
        
        if not html:
            break
            
        soup = BeautifulSoup(html, "html.parser")
        posters = soup.find_all("div", class_="film-poster")
        
        if not posters:
            # 더 이상 영화가 없으면 종료
            break
            
        for poster in posters:
            slug = poster.get("data-film-slug")
            if slug:
                slugs.append(slug)
                
        random_sleep(1.0, 2.5)
        page += 1
        
    return slugs

def crawl_reviews_for_movie(slug, existing_review_ids, all_data):
    """특정 영화의 리뷰를 페이지 끝까지 크롤링합니다."""
    # 먼저 영화 제목을 가져오기 위해 리뷰 1페이지 접근
    page = 1
    movie_title = None
    
    while True:
        if MAX_PAGES_PER_MOVIE and page > MAX_PAGES_PER_MOVIE:
            print(f"[{slug}] 설정된 최대 페이지({MAX_PAGES_PER_MOVIE})에 도달하여 크롤링을 중단합니다.")
            break
            
        url = f"https://letterboxd.com/film/{slug}/reviews/by/activity/page/{page}/"
        print(f"[{slug}] 리뷰 페이지 {page} 크롤링 중...")
        
        html = get_html(url)
        if not html:
            break
            
        soup = BeautifulSoup(html, "html.parser")
        
        # 첫 페이지에서 영화 제목 추출
        if page == 1 and not movie_title:
            title_tag = soup.find("span", class_="headline-1")
            if title_tag:
                movie_title = title_tag.text.strip()
            else:
                # 대안으로 다른 곳에서 제목 추출 시도
                alt_title = soup.find("a", href=f"/film/{slug}/")
                if alt_title:
                    movie_title = alt_title.text.strip()
                else:
                    movie_title = slug.replace("-", " ").title()
        
        # 리뷰 리스트 추출
        review_items = soup.find_all("li", class_="film-detail")
        
        if not review_items:
            # 더 이상 리뷰가 없음
            break
            
        new_reviews_count = 0
        
        for item in review_items:
            review_div = item.find("div", class_="film-detail-content")
            if not review_div:
                continue
                
            # 리뷰 고유 ID (중복 방지용)
            review_id = item.get("data-review-id", "")
            if not review_id:
                # data-review-id가 없으면 작성자 아이디를 추출해 임시 ID 생성
                author_link = review_div.find("a", class_="context")
                if author_link:
                    author_href = author_link.get("href", "")
                    author_name = author_href.strip("/").split("/")[-1] # /username/ 형태에서 추출
                    review_id = f"{slug}_{author_name}"
                    
            if review_id in existing_review_ids:
                # 이미 수집된 리뷰는 패스
                continue
                
            # 평점 추출
            rating_tag = review_div.find("span", class_="rating")
            rating = rating_tag.text.strip() if rating_tag else ""
            
            # 리뷰 텍스트 추출 (스포일러가 있는 경우나 접혀있는 텍스트 포함)
            body_text_div = review_div.find("div", class_="body-text")
            review_text = ""
            if body_text_div:
                paragraphs = body_text_div.find_all("p")
                review_text = "\n".join([p.text.strip() for p in paragraphs])
                
            # 텍스트가 비어있지 않은 경우만 저장
            if review_text:
                review_data = {
                    "review_id": review_id,
                    "movie_title": movie_title,
                    "movie_slug": slug,
                    "rating": rating,
                    "review": review_text
                }
                all_data.append(review_data)
                existing_review_ids.add(review_id)
                new_reviews_count += 1
                
        # I/O 병목 방지를 위해 10페이지 단위로 중간 저장
        if new_reviews_count > 0 and page % 10 == 0:
            save_data(all_data)
            
        random_sleep(2.0, 4.0)
        page += 1

    # 한 영화의 크롤링이 완전히 끝난 후 최종 저장
    save_data(all_data)

def main():
    print(f"=== Letterboxd {START_YEAR}~{END_YEAR}년도 개봉 영화 리뷰 크롤링 시작 ===")
    
    # 기존 데이터 불러오기
    all_data = load_existing_data()
    existing_review_ids = {item.get("review_id") for item in all_data if item.get("review_id")}
    
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\n[{year}년] 영화 목록 수집 시작...")
        slugs = get_movie_slugs_for_year(year)
        print(f"[{year}년] 총 {len(slugs)}개의 영화를 찾았습니다.")
        
        for idx, slug in enumerate(slugs, 1):
            print(f"\n>>> [{year}년] 영화 {idx}/{len(slugs)} : {slug} 리뷰 크롤링 시작...")
            crawl_reviews_for_movie(slug, existing_review_ids, all_data)
            
    print("\n=== 모든 크롤링 작업이 완료되었습니다. ===")

if __name__ == "__main__":
    main()
