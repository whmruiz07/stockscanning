# 🚀 選股神器｜Momentum Screener

一個基於 **Streamlit** 製作的美股動能掃描器，只需幾個按鈕就能：
- 找出動能最強的股票（S&P 500 / NASDAQ 100 / 自訂清單）
- 顯示 1/3/6 個月報酬、是否高於 50/200 日均線、RSI、ATR 等技術指標
- 自動計算 **止蝕價**（ATR×倍數 + 20日低）與 **止賺價**（R 倍數）

> **免安裝版**：本程式可部署到 [Streamlit Cloud](https://share.streamlit.io) 或 [Hugging Face Spaces](https://huggingface.co/spaces) 免費使用  
> **即時數據**：透過 [Yahoo Finance](https://finance.yahoo.com/) 抓取

---

## 🖥 本地運行

1. 下載專案  
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

### **方法 1：Streamlit Community Cloud**
1. Fork 或上傳此 Repo 到自己的 GitHub
2. 登入 [Streamlit Cloud](https://share.streamlit.io) → **New app**
3. 選擇：
   - Repository: 你的 Repo
   - Branch: main
   - Main file path: `momentum_screener_streamlit.py`
4. 點擊 **Deploy**  
   完成後會獲得一條可分享的網址

---

### **方法 2：Hugging Face Spaces**
1. 登入 [Hugging Face](https://huggingface.co)
2. **Create new Space** → 選擇 **Streamlit**
3. 上傳：
   - `momentum_screener_streamlit.py`（可改名 `app.py`）
   - `requirements.txt`
4. 儲存後自動建置並生成網址

---

## 📊 功能介紹
- **動能評分**：加權 1/3/6 個月報酬 + 均線位置 + 52 週高點距離 + RSI 區間
- **成交額過濾**：可設定最低日成交額（美元）
- **入場價設定**：可用現價或自訂
- **止蝕計算**：`max(20日低, 現價 - ATR×倍數)`
- **止賺計算**：以 R 倍數（1.5R / 2R / 3R）

---

## 📷 範例截圖
![screenshot](screenshot.png)

---

## 📜 授權
MIT License
