# 🚀 選股神器｜Momentum Screener

一個基於 **Streamlit** 製作的美股動能掃描器，幾個按鈕即可：
- 掃描 S&P 500 / NASDAQ 100 / 自訂股票清單
- 動能分數視覺化：🟢 強　🟡 中　🔴 弱
- 自動計算止蝕價（ATR×倍數 + 20日低）與止賺價（R 倍數）

---

## 🖥 本地運行
1. 下載專案檔案
2. 安裝依賴：
   ```bash
   pip install -r requirements.txt
   ```
3. 啟動：
   ```bash
   streamlit run momentum_screener_streamlit.py
   ```
4. 瀏覽器開啟 `http://localhost:8501`

---

## ☁ 雲端部署

### 方法 1：Streamlit Community Cloud
1. Fork 或上傳此 Repo 至 GitHub
2. 到 [Streamlit Cloud](https://share.streamlit.io) → New app
3. 選擇：
   - Repository: 你的 Repo
   - Branch: main
   - Main file path: `momentum_screener_streamlit.py`
4. 點 Deploy → 取得可分享的網址

### 方法 2：Hugging Face Spaces
1. 登入 [Hugging Face](https://huggingface.co)
2. Create new Space → 選擇 **Streamlit**
3. 上傳：
   - `momentum_screener_streamlit.py`（可改名 `app.py`）
   - `requirements.txt`
4. 系統自動建置並生成網址

---

## 📊 功能特色
- 動能評分：加權 1/3/6 個月報酬、均線位置、52 週高點距離、RSI 區間
- 成交額過濾：設定最低日成交額
- 入場價：現價 / 自訂價
- 止蝕：`max(20日低, 現價 - ATR×倍數)`
- 止賺：R 倍數（1.5R / 2R / 3R）

---

## 📷 範例截圖
![screenshot](screenshot.png)
