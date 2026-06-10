"""Fetch and format news data via newsapi.org."""

import json
import os

import requests
from dotenv import load_dotenv

from src.agents.agent_utils import log_api_request

BASE_URL = "https://newsapi.org/v2"


def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("NEWSAPI_API_KEY")
    if not api_key:
        raise ValueError(
            "NEWSAPI_API_KEY is not set. Add it to a .env file in the project root."
        )
    return api_key


def build_request_url(endpoint: str) -> str:
    return f"{BASE_URL}/{endpoint}"


def send_news_request(endpoint: str, params: dict) -> dict:
    log_api_request("newsapi.org", endpoint, params)
    api_key = load_api_key()
    request_params = {**params, "apiKey": api_key}
    response = requests.get(
        build_request_url(endpoint),
        params=request_params,
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def format_articles_response(payload: dict) -> str:
    if payload.get("status") != "ok":
        return json.dumps(
            {
                "error": payload.get("message", "NewsAPI request failed."),
                "code": payload.get("code"),
            },
            indent=2,
        )

    articles = []
    for article in payload.get("articles", []):
        articles.append(
            {
                "title": article.get("title"),
                "source": (article.get("source") or {}).get("name"),
                "author": article.get("author"),
                "published_at": article.get("publishedAt"),
                "url": article.get("url"),
                "description": article.get("description"),
            }
        )

    return json.dumps(
        {
            "total_results": payload.get("totalResults", len(articles)),
            "articles": articles,
        },
        indent=2,
    )


def search_news(query: str, page_size: int = 5) -> str:
    payload = send_news_request(
        "everything",
        {
            "q": query.strip(),
            "pageSize": page_size,
            "sortBy": "publishedAt",
            "language": "en",
        },
    )
    return format_articles_response(payload)


def get_top_headlines(
    country: str = "us",
    category: str | None = None,
    query: str | None = None,
    page_size: int = 5,
) -> str:
    params: dict = {
        "country": country.strip().lower(),
        "pageSize": page_size,
    }
    if category:
        params["category"] = category.strip().lower()
    if query:
        params["q"] = query.strip()

    payload = send_news_request("top-headlines", params)
    return format_articles_response(payload)
