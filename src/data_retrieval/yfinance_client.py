"""Fetch and format stock data via yfinance."""

import json

import yfinance as yf

from src.agents.agent_utils import log_api_request


def load_ticker(symbol: str) -> yf.Ticker:
    return yf.Ticker(symbol.strip().upper())


def fetch_quote_fields(ticker: yf.Ticker) -> dict:
    info = ticker.info or {}
    return {
        "symbol": info.get("symbol") or ticker.ticker,
        "name": info.get("shortName") or info.get("longName"),
        "currency": info.get("currency"),
        "exchange": info.get("exchange"),
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "previous_close": info.get("previousClose") or info.get("regularMarketPreviousClose"),
        "day_high": info.get("dayHigh") or info.get("regularMarketDayHigh"),
        "day_low": info.get("dayLow") or info.get("regularMarketDayLow"),
        "volume": info.get("volume") or info.get("regularMarketVolume"),
        "market_cap": info.get("marketCap"),
    }


def format_quote_response(quote: dict) -> str:
    if not quote.get("current_price"):
        return json.dumps(
            {
                "error": f"No quote data found for symbol '{quote.get('symbol')}'.",
                "hint": "Use a valid ticker such as AAPL, MSFT, or TSLA.",
            },
            indent=2,
        )
    return json.dumps(quote, indent=2, default=str)


def get_stock_quote(symbol: str) -> str:
    log_api_request("yahoo/yfinance", "get_stock_quote", {"symbol": symbol})
    ticker = load_ticker(symbol)
    quote = fetch_quote_fields(ticker)
    return format_quote_response(quote)


def fetch_history(ticker: yf.Ticker, period: str) -> list[dict]:
    history = ticker.history(period=period)
    if history.empty:
        return []
    rows = []
    for date, row in history.iterrows():
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 4),
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
                "close": round(float(row["Close"]), 4),
                "volume": int(row["Volume"]),
            }
        )
    return rows


def format_history_response(symbol: str, period: str, rows: list[dict]) -> str:
    if not rows:
        return json.dumps(
            {
                "error": f"No history found for symbol '{symbol}' with period '{period}'.",
                "hint": "Try period values like 5d, 1mo, 3mo, 6mo, 1y, 5y.",
            },
            indent=2,
        )
    return json.dumps(
        {
            "symbol": symbol.upper(),
            "period": period,
            "rows": rows,
        },
        indent=2,
    )


def get_stock_history(symbol: str, period: str = "1mo") -> str:
    log_api_request(
        "yahoo/yfinance",
        "get_stock_history",
        {"symbol": symbol, "period": period},
    )
    ticker = load_ticker(symbol)
    rows = fetch_history(ticker, period)
    return format_history_response(symbol, period, rows)
