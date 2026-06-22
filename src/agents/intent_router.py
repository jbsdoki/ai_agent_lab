"""Keyword-based routing for coordinator subagent selection.

Scans user prompts to decide which subagents should be enabled for a turn.
No LLM call is required.
"""

import re

FINANCE_SUBAGENT = "finance_subagent"
NEWS_SUBAGENT = "news_subagent"
WEB_SUBAGENT = "web_subagent"
SEC_SUBAGENT = "sec_subagent"
FILES_SUBAGENT = "files_subagent"
RAG_SUBAGENT = "rag_subagent"

URL_PATTERN = re.compile(r"https?://[^\s]+", re.IGNORECASE)

FINANCE_KEYWORDS = (
    "stock", "stocks", "share price", "stock price", "trading at",
    "trading", "ticker", "market cap", "yfinance", "nasdaq", "nyse", "quote",
)

NEWS_KEYWORDS = (
    "news", "headline", "headlines", "article", "articles",
    "breaking", "recent events", "top stories",
)

SEC_KEYWORDS = (
    "sec", "edgar", "filing", "filings", "10-k", "10-q", "8-k",
    "cik", "accession", "annual report", "quarterly report",
)

WEB_KEYWORDS = (
    "http://", "https://", "website", "web page", "webpage", "web site",
    "scrape", "fetch url", "summarize this link", "read this url", "open this link",
)

FILES_KEYWORDS = (
    "sandbox", "local file", "local files", "repository file",
    "repo file", "in this repo", "in the repo", "src/",
)

RAG_KEYWORDS = (
    "search documents", "find passage", "find passages", "semantic search",
    "what does the readme", "in the documentation", "vector search",
    "search the docs", "search the corpus", "search the project docs",
)

DEFAULT_SUBAGENTS = {FINANCE_SUBAGENT, NEWS_SUBAGENT}


def normalize_prompt(prompt: str) -> str:
    return prompt.strip().lower()


def match_keywords(prompt: str, keywords: tuple[str, ...]) -> bool:
    normalized = normalize_prompt(prompt)
    return any(keyword in normalized for keyword in keywords)


def contains_url(prompt: str) -> bool:
    return URL_PATTERN.search(prompt) is not None


def matches_finance_intent(prompt: str) -> bool:
    return match_keywords(prompt, FINANCE_KEYWORDS)


def matches_news_intent(prompt: str) -> bool:
    return match_keywords(prompt, NEWS_KEYWORDS)


def matches_sec_intent(prompt: str) -> bool:
    return match_keywords(prompt, SEC_KEYWORDS)


def matches_web_intent(prompt: str) -> bool:
    return match_keywords(prompt, WEB_KEYWORDS) or contains_url(prompt)


def matches_files_intent(prompt: str) -> bool:
    return match_keywords(prompt, FILES_KEYWORDS)


def matches_rag_intent(prompt: str) -> bool:
    return match_keywords(prompt, RAG_KEYWORDS)


def get_default_subagents() -> set[str]:
    return set(DEFAULT_SUBAGENTS)


def detect_subagent_intents(prompt: str) -> set[str]:
    intents: set[str] = set()

    if matches_finance_intent(prompt):
        intents.add(FINANCE_SUBAGENT)
    if matches_news_intent(prompt):
        intents.add(NEWS_SUBAGENT)
    if matches_sec_intent(prompt):
        intents.add(SEC_SUBAGENT)
    if matches_web_intent(prompt):
        intents.add(WEB_SUBAGENT)
    if matches_rag_intent(prompt):
        intents.add(RAG_SUBAGENT)
    if matches_files_intent(prompt):
        intents.add(FILES_SUBAGENT)

    if not intents:
        intents = get_default_subagents()

    return intents


def is_subagent_enabled(prompt: str, subagent_name: str) -> bool:
    return subagent_name in detect_subagent_intents(prompt)


def format_router_log_line(prompt: str, enabled_subagents: set[str]) -> str:
    ordered = sorted(enabled_subagents)
    return f"[ROUTER] enabled_subagents={ordered} prompt={prompt!r}"