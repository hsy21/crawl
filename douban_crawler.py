import csv
import json
import os
import random
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

START_YEAR = 2010
END_YEAR = 2026
END_MONTH = 5

COOKIE = ""

# 한 (태그/연도) 조합에서 가져올 최대 영화 수
MAX_MOVIES_PER_YEAR = 200

# 영화당 수집할 최대 리뷰(짧은 코멘트) 수
MAX_REVIEWS_PER_MOVIE = 10

MIN_DELAY = 3.0
MAX_DELAY = 6.0

OUTPUT_FILE = "douban_movies.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/json,*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,ko;q=0.8,en;q=0.7",
    "Referer": "https://movie.douban.com/",
}

# HTTP 세션

def build_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    if COOKIE:
        s.headers["Cookie"] = COOKIE
    return s


def polite_sleep():
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))


def get_with_retry(session, url, params=None, max_retries=3, is_json=False):
    """지연 + 재시도 포함 GET 요청."""
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, params=params, timeout=20)
            if resp.status_code == 200:
                return resp.json() if is_json else resp.text
            if resp.status_code in (403, 429):
                wait = 30 * attempt
                print(f"  [경고] {resp.status_code} 차단/제한. {wait}초 대기 후 재시도...")
                time.sleep(wait)
                continue
            print(f"  [오류] status={resp.status_code} url={url}")
        except requests.RequestException as e:
            print(f"  [예외] {e} (시도 {attempt}/{max_retries})")
            time.sleep(10 * attempt)
    return None


def fetch_movies_by_year(session, year):
    """해당 연도에 개봉한 영화 목록(id, 제목, 평점)을 반환."""
    movies = []
    api = "https://movie.douban.com/j/new_search_subjects"
    page_size = 20
    start = 0

    while start < MAX_MOVIES_PER_YEAR:
        params = {
            "sort": "R",          # R=평가많은순
            "range": "0,10",
            "tags": "电影",
            "start": start,
            "year_range": f"{year},{year}",
        }
        data = get_with_retry(session, api, params=params, is_json=True)
        polite_sleep()

        if not data or "data" not in data or not data["data"]:
            break

        for item in data["data"]:
            movies.append({
                "id": item.get("id"),
                "title": item.get("title"),
                "rate": item.get("rate"),   # 목록상의 평점
                "year": year,
                "url": item.get("url"),
            })

        if len(data["data"]) < page_size:
            break
        start += page_size

    return movies

# 영화 상세 페이지에서 짧은 리뷰(코멘트) 수집

def fetch_reviews(session, movie_id):
    """영화의 짧은 코멘트(리뷰)를 수집."""
    reviews = []
    url = f"https://movie.douban.com/subject/{movie_id}/comments"
    params = {"start": 0, "limit": 20, "status": "P", "sort": "new_score"}

    html = get_with_retry(session, url, params=params)
    polite_sleep()
    if not html:
        return reviews

    soup = BeautifulSoup(html, "html.parser")
    for node in soup.select("div.comment-item"):
        content = node.select_one("span.short")
        rating = node.select_one("span.rating")
        if content:
            reviews.append({
                "text": content.get_text(strip=True),
                "rating": rating["title"] if rating and rating.has_attr("title") else "",
            })
        if len(reviews) >= MAX_REVIEWS_PER_MOVIE:
            break
    return reviews


def init_csv(path):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["movie_id", "title", "year", "rate", "review_rating", "review_text"])


def append_movie(path, movie, reviews):
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if reviews:
            for r in reviews:
                writer.writerow([
                    movie["id"], movie["title"], movie["year"],
                    movie["rate"], r["rating"], r["text"],
                ])
        else:
            # 리뷰가 없어도 영화 정보는 저장
            writer.writerow([movie["id"], movie["title"], movie["year"], movie["rate"], "", ""])

def main():
    session = build_session()
    init_csv(OUTPUT_FILE)

    seen = set()  # 중복 영화 방지

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\n===== {year}년 영화 수집 시작 =====")
        movies = fetch_movies_by_year(session, year)
        print(f"  {year}년: {len(movies)}편 발견")

        for i, movie in enumerate(movies, 1):
            if not movie["id"] or movie["id"] in seen:
                continue
            seen.add(movie["id"])

            # 2026년은 5월까지만 (정확한 월 필터는 상세페이지의 개봉일 필요 → 연도
            print(f"  [{year}] ({i}/{len(movies)}) {movie['title']} (평점 {movie['rate']})")
            reviews = fetch_reviews(session, movie["id"])
            append_movie(OUTPUT_FILE, movie, reviews)

    print(f"\n완료! 결과 파일: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
