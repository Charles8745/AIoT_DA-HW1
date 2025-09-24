# AIoT_DA-HW1：簡單線性迴歸互動展示

本專案以 Python 實作簡單線性迴歸，並依 CRISP-DM 流程開發，提供互動式網頁介面（Streamlit），讓使用者可調整參數並即時觀察資料、回歸線與評估指標。

---

## 目錄
- [專案簡介](#專案簡介)
- [功能特色](#功能特色)
- [安裝教學](#安裝教學)
- [執行方式](#執行方式)
- [參數說明](#參數說明)
- [CRISP-DM 流程紀錄](#crisp-dm-流程紀錄)
- [常見問題](#常見問題)

---

## 專案簡介
- 以 Python + Streamlit 實作互動式線性迴歸展示。
- 可自訂斜率 a、截距 b、雜訊、資料點數等參數。
- 即時顯示資料分布、回歸線與模型評估指標。

## 功能特色
- 產生可調參數的線性資料集
- 線性迴歸模型訓練、預測、評估
- 互動式網頁介面（Streamlit）
- 完整 CRISP-DM 步驟紀錄

## 安裝教學
1. **安裝 Python 3.8 以上版本**（建議 3.10）
2. **下載本專案**
   ```bash
   git clone https://github.com/Charles8745/AIoT_DA-HW1.git
   cd AIoT_DA-HW1
   ```
3. **建立虛擬環境（建議）**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
4. **安裝必要套件**
   ```bash
   pip install -r requirements.txt
   ```
   若無 requirements.txt，可手動安裝：
   ```bash
   pip install numpy pandas scikit-learn matplotlib streamlit
   ```

## 執行方式
1. 於終端機進入專案資料夾
2. 執行下列指令啟動網頁介面：
   ```bash
   streamlit run app.py
   ```
3. 預設會自動開啟瀏覽器，或於網址列輸入 http://localhost:8501

## 參數說明
- **斜率 a**：決定資料線性趨勢的斜率
- **截距 b**：線性方程式的截距
- **雜訊標準差**：資料隨機擾動程度
- **資料點數**：產生的資料數量
- **隨機種子**：確保每次產生資料一致

## CRISP-DM 流程紀錄
- 詳細開發紀錄請見 `log.md`
- 專案規劃與流程請見 `project_plan_AImodify.md`

## 常見問題
- 若遇到套件安裝失敗，請確認 Python 版本與網路連線。
- 若無法啟動 Streamlit，請確認已安裝所有必要套件。
- 其他問題請於 GitHub issue 留言。

---

> 本專案歡迎教學、研究與自學使用。
