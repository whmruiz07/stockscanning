import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="美股動能選股神器", layout="wide")

@st.cache_data(show_spinner=False)
def get_universe_tickers(kind: str):
    # NASDAQ-100 fallback list (as of 2025-08)
    nas100 = [
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "PEP", "COST",
        "AVGO", "ADBE", "NFLX", "AMD", "CSCO", "INTC", "LIN", "TMUS", "TXN", "AMGN",
        "QCOM", "HON", "SBUX", "INTU", "CHTR", "PDD", "AMAT", "BKNG", "MU", "LRCX",
        "PANW", "GILD", "VRTX", "REGN", "ADP", "MDLZ", "MELI", "CSX", "KDP", "MAR",
        "MNST", "ABNB", "FTNT", "KLAC", "ORLY", "SNPS", "ADSK", "PAYX", "CTAS", "ROP",
        "WDAY", "ODFL", "PCAR", "ROST", "CDNS", "NXPI", "EXC", "KHC", "DLTR", "EA",
        "VRSK", "XEL", "IDXX", "CPRT", "AZN", "BIDU", "BKR", "ZS", "MRVL", "CRWD",
        "DDOG", "LCID", "LULU", "TEAM", "SPLK", "DOCU", "EBAY", "OKTA", "ANSS", "ALGN",
        "VRSN", "SWKS", "FAST", "SIRI", "MTCH", "TTD", "BIIB", "JD", "CHKP", "ILMN",
        "CDW", "CEG", "FANG", "WBD", "TTWO", "RIVN", "ZM", "SGEN", "TSCO", "ZSAN"
    ]

    if kind == "S&P 500":
        try:
            return sorted([t.replace(".", "-") for t in yf.tickers_sp500()])
        except Exception:
            st.warning("⚠️ 無法取得 S&P 500 清單，請改用自訂清單")
            return []
    elif kind == "NASDAQ 100":
        return sorted(nas100)
    else:
        return []

def calculate_momentum(ticker):
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if len(df) < 50:
            return None
        df["Return_20d"] = df["Adj Close"].pct_change(20)
        df["Return_60d"] = df["Adj Close"].pct_change(60)
        df["Return_120d"] = df["Adj Close"].pct_change(120)
        score = df[["Return_20d", "Return_60d", "Return_120d"]].iloc[-1].mean()
        last_price = df["Adj Close"].iloc[-1]
        stop_loss = last_price * 0.9
        take_profit = last_price * 1.1
        return {
            "Ticker": ticker,
            "Last Price": last_price,
            "20d %": df["Return_20d"].iloc[-1],
            "60d %": df["Return_60d"].iloc[-1],
            "120d %": df["Return_120d"].iloc[-1],
            "Score": score,
            "Stop Loss": stop_loss,
            "Take Profit": take_profit
        }
    except Exception:
        return None

st.sidebar.header("📊 選股設定")
universe_choice = st.sidebar.selectbox("選擇股票池", ["S&P 500", "NASDAQ 100", "自訂（貼上代號）"])
if universe_choice == "自訂（貼上代號）":
    custom_list = st.sidebar.text_area("輸入代號（用逗號分隔）", "AAPL,MSFT,TSLA")
    tickers = [t.strip().upper() for t in custom_list.split(",") if t.strip()]
else:
    tickers = get_universe_tickers(universe_choice)

if not tickers:
    st.stop()

st.sidebar.write(f"股票數量：{len(tickers)}")
run_button = st.sidebar.button("🚀 開始分析")

if run_button:
    results = []
    for t in tickers:
        r = calculate_momentum(t)
        if r:
            results.append(r)
    if results:
        df_results = pd.DataFrame(results)
        df_results.sort_values("Score", ascending=False, inplace=True)
        st.success(f"✅ 分析完成！共 {len(df_results)} 檔")
        st.dataframe(df_results.style.background_gradient(cmap="RdYlGn", subset=["Score", "20d %", "60d %", "120d %"]))
    else:
        st.warning("沒有符合條件的股票")