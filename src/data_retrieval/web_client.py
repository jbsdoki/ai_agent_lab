"""Fetch and extract readable text from allowlisted HTTPS pages."""

import json
import os
import time
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from src.agents.agent_utils import log_api_request

DEFAULT_ALLOWED_DOMAINS = "sec.gov,apple.com,techcrunch.com,news.ycombinator.com"
DEFAULT_USER_AGENT = "AI_Agent_Lab/1.0 WebFetcher"
MAX_TEXT_LENGTH = 6000
MIN_REQUEST_INTERVAL_SECONDS = 1.0

_last_request_at: float = 0.0


def load_allowed_domains() -> list[str]:
    load_dotenv()
    raw = os.getenv("WEB_ALLOWED_DOMAINS", DEFAULT_ALLOWED_DOMAINS)
    return [domain.strip().lower() for domain in raw.split(",") if domain.strip()]


def wait_for_rate_limit() -> None:
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
        time.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)
    _last_request_at = time.monotonic()


def parse_https_url(url: str) -> tuple[str, str]:
    parsed = urlparse(url.strip())
    if parsed.scheme != "https":
        raise ValueError("Only HTTPS URLs are allowed.")
    if not parsed.netloc:
        raise ValueError("URL must include a host.")
    return parsed.geturl(), parsed.netloc.lower()


def normalize_host(host: str) -> str:
    lowered = host.lower()
    if lowered.startswith("www."):
        return lowered[4:]
    return lowered


def is_host_allowed(host: str, allowed_domains: list[str]) -> bool:
    normalized_host = normalize_host(host)
    for domain in allowed_domains:
        if normalized_host == domain or normalized_host.endswith(f".{domain}"):
            return True
    return False


def validate_url(url: str) -> tuple[str, str]:
    validated_url, host = parse_https_url(url)
    allowed_domains = load_allowed_domains()
    if not is_host_allowed(host, allowed_domains):
        raise ValueError(
            f"Host '{host}' is not allowlisted. Allowed domains: {', '.join(allowed_domains)}"
        )
    return validated_url, host


def build_request_headers() -> dict[str, str]:
    return {"User-Agent": DEFAULT_USER_AGENT}


def fetch_page_html(url: str) -> str:
    wait_for_rate_limit()
    log_api_request("web", url, {})
    response = requests.get(
        url,
        headers=build_request_headers(),
        timeout=20,
    )
    response.raise_for_status()
    return response.text


def remove_boilerplate_tags(soup: BeautifulSoup) -> None:
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()


def extract_title(soup: BeautifulSoup) -> str | None:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return None


def extract_meta_description(soup: BeautifulSoup) -> str | None:
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        return str(meta["content"]).strip()
    return None


def extract_paragraph_text(soup: BeautifulSoup) -> str:
    paragraphs: list[str] = []
    for paragraph in soup.find_all("p"):
        text = paragraph.get_text(" ", strip=True)
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def truncate_text(text: str, max_length: int = MAX_TEXT_LENGTH) -> tuple[str, bool]:
    if len(text) <= max_length:
        return text, False
    return text[:max_length] + "...", True


def extract_page_text(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    remove_boilerplate_tags(soup)
    body_text, truncated = truncate_text(extract_paragraph_text(soup))
    return {
        "title": extract_title(soup),
        "description": extract_meta_description(soup),
        "text": body_text,
        "text_length": len(body_text),
        "truncated": truncated,
    }


def format_error_response(message: str, hint: str | None = None) -> str:
    payload: dict = {"error": message}
    if hint:
        payload["hint"] = hint
    return json.dumps(payload, indent=2)


def format_fetch_response(url: str, host: str, extracted: dict) -> str:
    return json.dumps(
        {
            "url": url,
            "host": host,
            **extracted,
        },
        indent=2,
    )


def fetch_url(url: str) -> str:
    try:
        validated_url, host = validate_url(url)
        html = fetch_page_html(validated_url)
        extracted = extract_page_text(html)
    except ValueError as exc:
        return format_error_response(
            str(exc),
            "Use an HTTPS URL on an allowlisted domain.",
        )
    except requests.HTTPError as exc:
        return format_error_response(f"Page request failed: {exc}")
    except requests.RequestException as exc:
        return format_error_response(f"Page request failed: {exc}")

    return format_fetch_response(validated_url, host, extracted)
