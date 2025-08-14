
# Streamlit Momentum Screener — "選股神器 (Beta)"
# Author: ChatGPT for Homan
# Purpose: Few clicks to find high-momentum U.S. stocks and suggest stop-loss / take-profit levels.

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import math

# External data
import yfinance as yf

# ---------------------------
# Utility: technicals
# ---------------------------
def sma(series, window):
    return series.rolling(window).mean()

def ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def rsi(series, period: int = 14):
    # Wilder's RSI
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def atr(high, low, close, period: int = 14):
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def pct_change(series, periods):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (series / series.shift(periods) - 1.0) * 100.0

def safe_div(a, b):
    try:
        return float(a) / float(b) if float(b) != 0 else np.nan
    except Exception:
        return np.nan

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="選股神器｜Momentum Screener", layout="wide")
st.title("🚀 選股神器（Momentum Screener）")
st.caption("幾個按鈕找動能強的美股，並自動給出止賺/止蝕價格。")

# Sidebar — universe & params
with st.sidebar:
    st.header("⚙️ 參數設定")
    universe = st.selectbox("股票範圍", ["S&P 500", "NASDAQ 100", "自訂（貼上代號，逗號或空白分隔）"])
    lookback_1 = st.selectbox("短期動能（天）", [10, 20, 21, 30], index=2)
    lookback_3 = st.selectbox("中期動能（天）", [42, 60, 63, 90], index=2)
    lookback_6 = st.selectbox("長期動能（天）", [126, 180, 200, 252], index=0)
    min_dollar_vol = st.number_input("最低平均日成交額（美元）", min_value=0, value=2_000_000, step=500_000, help="以近20日平均成交額（成交量*收盤）計。")
    top_n = st.slider("顯示前 N 名", 5, 50, 15, step=5)
    entry_mode = st.radio("入場價格", ["使用最新收盤價", "自訂入場價"])
    custom_entry = st.number_input("自訂入場價（當選單股時才會用）", value=0.0, min_value=0.0, step=0.01, format="%.2f")
    risk_multiple_tp = st.select_slider("止賺倍數（R）", options=[1.5, 2.0, 2.5, 3.0], value=2.0)
    atr_k = st.select_slider("ATR 倍數止蝕", options=[1.5, 2.0, 2.5, 3.0], value=2.0)
    min_price = st.number_input("最低股價（過濾低價股）", min_value=0.0, value=5.0, step=1.0)
    run_scan = st.button("🔍 一鍵掃描")

# Helper: get tickers by universe
@st.cache_data(show_spinner=False)
def get_universe_tickers(kind: str):
    if kind == "S&P 500":
        try:
            sp = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
            return sorted(sp["Symbol"].unique().tolist())
        except Exception:
            st.warning("無法取得 S&P 500 清單，請改用自訂。")
            return []
    elif kind == "NASDAQ 100":
        try:
            nq = pd.read_html("https://en.wikipedia.org/wiki/NASDAQ-100")[3]
            syms = nq["Ticker"].dropna().astype(str).str.replace(".", "-", regex=False).tolist()
            return sorted(syms)
        except Exception:
            st.warning("無法取得 NASDAQ 100 清單，請改用自訂。")
            return []
    else:
        return []

# Custom tickers input
custom_tickers = ""
if universe == "自訂（貼上代號，逗號或空白分隔）":
    custom_tickers = st.text_area("貼上代號（例：AAPL, MSFT, NVDA 或換行分隔）", height=100)

# Fetch data
@st.cache_data(show_spinner=True, ttl=3600)
def fetch_history(tickers, period="400d"):
    data = yf.download(tickers=tickers, period=period, auto_adjust=True, threads=True, group_by="ticker", progress=False)
    return data

def parse_custom_tickers(raw: str):
    if not raw:
        return []
    parts = [p.strip().upper() for p in raw.replace("\n", " ").replace("\t", " ").replace(";", " ").replace("|"," ").split(" ") if p.strip()]
    # Also split by comma
    out = []
    for item in parts:
        out.extend([x.strip() for x in item.split(",") if x.strip()])
    # Remove duplicates & invalids
    out = [x.replace(".", "-") for x in out if x.isascii()]
    return sorted(list(dict.fromkeys(out)))

def average_dollar_volume(close, volume, window=20):
    return (close * volume).rolling(window).mean()

def score_row(row):
    # Higher is better
    score = 0.0
    # Momentum weights
    score += 0.4 * row.get("ret_%dD" % lookback_1, 0)
    score += 0.35 * row.get("ret_%dD" % lookback_3, 0)
    score += 0.25 * row.get("ret_%dD" % lookback_6, 0)
    # Trend filters
    score += 5.0 if row.get("above_50") else 0.0
    score += 5.0 if row.get("above_200") else 0.0
    # Near 52W high
    dist_high = row.get("dist_52w_high_pct", np.nan)
    if not math.isnan(dist_high):
        if dist_high >= -3:  # at or within 3% of 52w high
            score += 5.0
        elif dist_high >= -10:
            score += 2.5
    # RSI sweet spot
    rsi_v = row.get("rsi14", np.nan)
    if not math.isnan(rsi_v):
        if 55 <= rsi_v <= 75:
            score += 3.0
        elif 50 <= rsi_v < 55:
            score += 1.0
        elif rsi_v > 80:  # possibly overbought—slightly penalize
            score -= 2.0
    return score

def compute_levels(close_series, atr_series, atr_mult=2.0):
    last_close = float(close_series.iloc[-1])
    last_atr = float(atr_series.iloc[-1])
    # Swing low: last 20-day low
    swing_low = float(close_series.rolling(20).min().iloc[-1])
    atr_stop = last_close - atr_mult * last_atr
    # Use the tighter stop (higher price) to control risk
    stop = max(atr_stop, swing_low)
    risk = max(last_close - stop, 0.01)  # avoid zero
    tp1 = last_close + risk  # 1R
    tp2 = last_close + 2 * risk  # 2R
    tp3 = last_close + 3 * risk  # 3R
    return last_close, stop, tp1, tp2, tp3, risk

def format_price(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return ""

def run_screen(tickers):
    if not tickers:
        st.info("⚠️ 沒有可用代號。請更換股票範圍或貼上自訂清單。")
        return

    raw = fetch_history(tickers)
    # Normalize to MultiIndex [Ticker, Field]
    records = []
    for tkr in tickers:
        try:
            df = raw[tkr].dropna().copy()
            if df.empty:
                continue

            # Basic columns
            df["SMA50"] = sma(df["Close"], 50)
            df["SMA200"] = sma(df["Close"], 200)
            df["RSI14"] = rsi(df["Close"], 14)
            df["ATR14"] = atr(df["High"], df["Low"], df["Close"], 14)
            df["ADVol20"] = average_dollar_volume(df["Close"], df["Volume"], 20)

            # Momentum returns
            df[f"RET_{lookback_1}D"] = pct_change(df["Close"], lookback_1)
            df[f"RET_{lookback_3}D"] = pct_change(df["Close"], lookback_3)
            df[f"RET_{lookback_6}D"] = pct_change(df["Close"], lookback_6)

            # 52-week metrics
            df["HIGH_52W"] = df["Close"].rolling(252).max()
            df["dist_52w_high_pct"] = (df["Close"] / df["HIGH_52W"] - 1.0) * 100.0

            last = df.iloc[-1]
            if (last["ADVol20"] or 0) < min_dollar_vol:
                continue
            if last["Close"] < min_price:
                continue

            row = {
                "Ticker": tkr,
                "Price": last["Close"],
                "ret_%dD" % lookback_1: last[f"RET_{lookback_1}D"],
                "ret_%dD" % lookback_3: last[f"RET_{lookback_3}D"],
                "ret_%dD" % lookback_6: last[f"RET_{lookback_6}D"],
                "above_50": bool(last["Close"] > last["SMA50"]),
                "above_200": bool(last["Close"] > last["SMA200"]),
                "rsi14": last["RSI14"],
                "dist_52w_high_pct": last["dist_52w_high_pct"],
                "atr14": last["ATR14"],
                "avg_dollar_vol_20d": last["ADVol20"],
            }

            # Levels (based on last close)
            close, stop, tp1, tp2, tp3, risk = compute_levels(df["Close"], df["ATR14"], atr_mult=atr_k)
            row.update({
                "stop_loss": stop,
                "tp_1R": tp1,
                "tp_2R": tp2,
                "tp_3R": tp3,
                "R(risk)": risk,
            })

            row["score"] = score_row(row)
            records.append(row)
        except Exception as e:
            # Skip ticker on error
            continue

    if not records:
        st.warning("沒有符合條件的股票。試著放寬成交額或價格門檻。")
        return

    out = pd.DataFrame(records).sort_values("score", ascending=False).reset_index(drop=True)
    st.subheader("📈 掃描結果（Momentum Ranking）")
    st.dataframe(
        out.head(top_n).assign(
            Price=lambda d: d["Price"].map(format_price),
            rsi14=lambda d: d["rsi14"].round(1),
            dist_52w_high_pct=lambda d: d["dist_52w_high_pct"].round(2),
            atr14=lambda d: d["atr14"].map(format_price),
            stop_loss=lambda d: d["stop_loss"].map(format_price),
            tp_2R=lambda d: d["tp_2R"].map(format_price),
            tp_3R=lambda d: d["tp_3R"].map(format_price),
            **{f"ret_{lookback_1}D": lambda d: d[f"ret_{lookback_1}D"].round(1),
               f"ret_{lookback_3}D": lambda d: d[f"ret_{lookback_3}D"].round(1),
               f"ret_{lookback_6}D": lambda d: d[f"ret_{lookback_6}D"].round(1)}
        ),
        use_container_width=True,
        hide_index=True
    )

    # Details — select one ticker
    st.markdown("---")
    sel = st.selectbox("查看詳情（繪圖 & 價位）", out.head(top_n)["Ticker"].tolist())
    if sel:
        df = fetch_history(sel, period="400d")
        df = df[sel].dropna().copy()
        df["SMA20"] = sma(df["Close"], 20)
        df["SMA50"] = sma(df["Close"], 50)
        df["SMA200"] = sma(df["Close"], 200)
        df["ATR14"] = atr(df["High"], df["Low"], df["Close"], 14)

        last_close, stop, tp1, tp2, tp3, risk = compute_levels(df["Close"], df["ATR14"], atr_mult=atr_k)

        if entry_mode == "自訂入場價" and custom_entry > 0:
            entry = custom_entry
            r = max(entry - stop, 0.01)
            tp = entry + risk_multiple_tp * r
        else:
            entry = last_close
            r = max(entry - stop, 0.01)
            tp = entry + risk_multiple_tp * r

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("現價", format_price(last_close))
        c2.metric("止蝕", format_price(stop))
        c3.metric(f"止賺（{risk_multiple_tp}R）", format_price(tp))
        c4.metric("ATR(14)", format_price(df["ATR14"].iloc[-1]))

        st.line_chart(df[["Close", "SMA20", "SMA50", "SMA200"]])

        st.caption("建議：以上止蝕為 `max(20日低, 現價 - ATR×倍數)`；止賺以 R 倍數計算。請自行依風險承受度調整。")

# Main flow
if run_scan:
    if universe == "自訂（貼上代號，逗號或空白分隔）":
        tickers = parse_custom_tickers(custom_tickers)
    else:
        tickers = get_universe_tickers(universe)
    # yfinance expects space-joined string or list
    run_screen(tickers)
else:
    st.info("在左側選擇股票範圍與條件後，按 **🔍 一鍵掃描** 開始。")
    st.markdown(
        """
        **方法說明**
        - 動能分數以 1/3/6 個月報酬、是否高於 50/200 日均線、是否接近 52 週新高、RSI 區間等加權組合而成。
        - 風控：止蝕取 **max(20日低, 價格-ATR×倍數)**；止賺以 **R 倍數**（例如 2R、3R）。
        - 可用「自訂入場價」來測試不同進場點的止賺/止蝕。
        """
    )
