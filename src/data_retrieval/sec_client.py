"""Fetch and format SEC EDGAR data via data.sec.gov."""

import json
import os
import time

import requests
from dotenv import load_dotenv

from src.agents.agent_utils import log_api_request

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_BASE_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
MIN_REQUEST_INTERVAL_SECONDS = 0.11

_ticker_map_cache: dict[str, dict] | None = None
_last_request_at: float = 0.0


def load_sec_headers() -> dict[str, str]:
    load_dotenv()
    user_agent = os.getenv("SEC_USER_AGENT")
    if not user_agent:
        raise ValueError(
            "SEC_USER_AGENT is not set. Add it to a .env file in the project root."
        )
    return {"User-Agent": user_agent}


def wait_for_rate_limit() -> None:
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
        time.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)
    _last_request_at = time.monotonic()


def fetch_json(url: str) -> dict:
    wait_for_rate_limit()
    log_api_request("sec.gov", url, {})
    headers = load_sec_headers()
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def pad_cik(cik: int | str) -> str:
    return str(cik).zfill(10)


def build_ticker_index(payload: dict) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for entry in payload.values():
        ticker = str(entry["ticker"]).upper()
        index[ticker] = {
            "ticker": ticker,
            "cik_str": int(entry["cik_str"]),
            "title": entry.get("title", ""),
        }
    return index


def load_ticker_map() -> dict[str, dict]:
    global _ticker_map_cache
    if _ticker_map_cache is None:
        payload = fetch_json(TICKER_MAP_URL)
        _ticker_map_cache = build_ticker_index(payload)
    return _ticker_map_cache


def resolve_ticker_entry(symbol: str) -> dict:
    ticker_map = load_ticker_map()
    normalized = symbol.strip().upper()
    if normalized not in ticker_map:
        raise ValueError(
            f"No SEC ticker match for '{symbol}'. Use a valid US ticker such as AAPL."
        )
    return ticker_map[normalized]


def resolve_cik(cik_or_ticker: str) -> str:
    value = cik_or_ticker.strip()
    if value.isdigit():
        return pad_cik(value)
    return pad_cik(resolve_ticker_entry(value)["cik_str"])


def build_submissions_url(cik: str) -> str:
    return SUBMISSIONS_BASE_URL.format(cik=cik)


def build_report_url(cik: str, accession_number: str, primary_document: str) -> str:
    cik_numeric = str(int(cik))
    accession_path = accession_number.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_numeric}/{accession_path}/{primary_document}"
    )


def format_error_response(message: str, hint: str | None = None) -> str:
    payload: dict = {"error": message}
    if hint:
        payload["hint"] = hint
    return json.dumps(payload, indent=2)


def format_cik_response(entry: dict) -> str:
    return json.dumps(
        {
            "symbol": entry["ticker"],
            "cik": pad_cik(entry["cik_str"]),
            "company_name": entry["title"],
        },
        indent=2,
    )


def format_submissions_response(submissions: dict) -> str:
    return json.dumps(
        {
            "cik": pad_cik(submissions.get("cik", "")),
            "name": submissions.get("name"),
            "tickers": submissions.get("tickers", []),
            "exchanges": submissions.get("exchanges", []),
            "sic": submissions.get("sic"),
            "sicDescription": submissions.get("sicDescription"),
            "fiscalYearEnd": submissions.get("fiscalYearEnd"),
        },
        indent=2,
    )


def extract_recent_filing_rows(
    submissions: dict,
    form_type: str | None,
    limit: int,
) -> list[dict]:
    recent = (submissions.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    filing_dates = recent.get("filingDate") or []
    accession_numbers = recent.get("accessionNumber") or []
    primary_documents = recent.get("primaryDocument") or []

    cik = pad_cik(submissions.get("cik", ""))
    normalized_form = form_type.strip().upper() if form_type else None
    rows: list[dict] = []

    for form, filing_date, accession_number, primary_document in zip(
        forms,
        filing_dates,
        accession_numbers,
        primary_documents,
        strict=False,
    ):
        if normalized_form and str(form).upper() != normalized_form:
            continue
        rows.append(
            {
                "form": form,
                "filingDate": filing_date,
                "accessionNumber": accession_number,
                "primaryDocument": primary_document,
                "reportUrl": build_report_url(cik, accession_number, primary_document),
            }
        )
        if len(rows) >= limit:
            break

    return rows


def format_filings_response(
    symbol: str,
    cik: str,
    company_name: str,
    filings: list[dict],
    form_type: str | None,
) -> str:
    if not filings:
        message = f"No recent filings found for '{symbol}'."
        if form_type:
            message = f"No recent '{form_type}' filings found for '{symbol}'."
        return format_error_response(
            message,
            "Try another form type such as 10-K, 10-Q, or 8-K.",
        )

    return json.dumps(
        {
            "symbol": symbol.upper(),
            "cik": cik,
            "company_name": company_name,
            "form_type_filter": form_type,
            "filings": filings,
        },
        indent=2,
    )


def lookup_cik(symbol: str) -> str:
    try:
        entry = resolve_ticker_entry(symbol)
    except ValueError as exc:
        return format_error_response(str(exc))
    return format_cik_response(entry)


def get_company_submissions(cik_or_ticker: str) -> str:
    try:
        cik = resolve_cik(cik_or_ticker)
        submissions = fetch_json(build_submissions_url(cik))
    except ValueError as exc:
        return format_error_response(str(exc))
    except requests.HTTPError as exc:
        return format_error_response(f"SEC submissions request failed: {exc}")
    return format_submissions_response(submissions)


def get_recent_filings(
    cik_or_ticker: str,
    form_type: str | None = None,
    limit: int = 5,
) -> str:
    try:
        if cik_or_ticker.strip().isdigit():
            entry = None
            cik = resolve_cik(cik_or_ticker)
        else:
            entry = resolve_ticker_entry(cik_or_ticker)
            cik = pad_cik(entry["cik_str"])

        submissions = fetch_json(build_submissions_url(cik))
        tickers = submissions.get("tickers") or []
        if entry:
            symbol = entry["ticker"]
        elif tickers:
            symbol = str(tickers[0]).upper()
        else:
            symbol = cik
        company_name = submissions.get("name") or (entry or {}).get("title", "")
        filings = extract_recent_filing_rows(submissions, form_type, limit)
    except ValueError as exc:
        return format_error_response(str(exc))
    except requests.HTTPError as exc:
        return format_error_response(f"SEC filings request failed: {exc}")

    return format_filings_response(symbol, cik, company_name, filings, form_type)
