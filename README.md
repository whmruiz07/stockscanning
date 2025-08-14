# 美股動能選股神器

這是一個用 Streamlit 製作的簡易美股動能篩選器，可分析 S&P 500、NASDAQ 100 或自訂清單的股票，並計算 20d、60d、120d 報酬率與平均動能分數，提供止損與止賺參考。

## 功能
- 支援 S&P 500（透過 yfinance 內建清單）
- 支援 NASDAQ-100（內建 fallback 清單，避免被封鎖）
- 支援自訂股票代號
- 自動計算止損（-10%）與止賺（+10%）參考價
- 動能分數色階顯示（綠=強、紅=弱）

## 安裝與執行
```bash
pip install -r requirements.txt
streamlit run momentum_screener_streamlit.py
```