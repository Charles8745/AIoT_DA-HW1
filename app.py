# ====== 所有 import 統一於最上方 ======
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ====== Streamlit 主程式 ======
def main():
    # 現代化配置
    st.set_page_config(
        layout="wide", 
        page_title="Linear Regression Interactive Demo",
        page_icon="📊",
        initial_sidebar_state="expanded"
    )
    
    # Glassmorphism CSS 樣式 - 白色基底
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* 主背景 - 白色漸變 */
        html, body, [class*="css"] {
            font-family: 'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #f8faff 0%, #ffffff 50%, #f0f4ff 100%);
        }
        
        /* 主容器 - Glassmorphism 效果 */
        .main .block-container {
            padding: 2rem;
            max-width: 1400px;
            background: rgba(255, 255, 255, 0.25);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 20px;
            margin: 1rem auto;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        /* 側邊欄 Glassmorphism - 修正可見性問題 */
        .css-1d391kg, section[data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.2) !important;
            backdrop-filter: blur(15px);
            border-right: 1px solid rgba(255, 255, 255, 0.3);
        }
        
        /* 側邊欄展開/收合按鈕 - 確保可見 */
        button[data-testid="collapsedControl"] {
            background: rgba(52, 152, 219, 0.9) !important;
            backdrop-filter: blur(15px) !important;
            border: 3px solid rgba(255, 255, 255, 0.8) !important;
            border-radius: 50% !important;
            color: white !important;
            box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4) !important;
            transition: all 0.3s ease !important;
            width: 50px !important;
            height: 50px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            position: fixed !important;
            top: 1.5rem !important;
            left: 1.5rem !important;
            z-index: 999999 !important;
            visibility: visible !important;
            opacity: 1 !important;
            font-size: 1.2rem !important;
            font-weight: bold !important;
        }
        
        button[data-testid="collapsedControl"]:hover {
            background: rgba(52, 152, 219, 1) !important;
            border-color: rgba(255, 255, 255, 1) !important;
            transform: scale(1.15) !important;
            box-shadow: 0 8px 25px rgba(52, 152, 219, 0.6) !important;
        }
        
        /* 側邊欄內容樣式 */
        .css-1lcbmhc, .css-17eq0hr {
            background: transparent;
        }
        
        /* Slider 容器 */
        .stSlider > div {
            background: rgba(255, 255, 255, 0.4);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.5);
            border-radius: 15px;
            padding: 1.2rem;
            margin: 0.8rem 0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        }
        
        /* 按鈕樣式 */
        .stButton > button {
            background: rgba(255, 255, 255, 0.3);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.5);
            color: #2c3e50;
            border-radius: 25px;
            padding: 0.7rem 2rem;
            font-size: 1rem;
            font-weight: 600;
            margin: 0.8rem 0;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        .stButton > button:hover {
            background: rgba(255, 255, 255, 0.5);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        }
        
        /* 主要按鈕 */
        div[data-testid="stButton"] button[kind="primary"] {
            background: linear-gradient(135deg, rgba(52, 152, 219, 0.8), rgba(155, 89, 182, 0.8));
            color: white;
            border: none;
        }
        
        div[data-testid="stButton"] button[kind="primary"]:hover {
            background: linear-gradient(135deg, rgba(52, 152, 219, 0.9), rgba(155, 89, 182, 0.9));
        }
        
        /* 指標卡片 */
        .metric-glass {
            background: rgba(255, 255, 255, 0.3);
            backdrop-filter: blur(15px);
            border: 1px solid rgba(255, 255, 255, 0.4);
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            margin: 1rem 0;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        
        .metric-glass:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 35px rgba(0, 0, 0, 0.15);
        }
        
        /* 標題樣式 */
        h1, h2, h3 {
            color: #2c3e50;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        
        /* 數值輸入框 */
        .stNumberInput > div > div > input {
            background: rgba(255, 255, 255, 0.4);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.5);
            border-radius: 10px;
            color: #2c3e50;
        }
        
        /* 選擇框樣式 */
        .stSelectbox > div > div {
            background: rgba(255, 255, 255, 0.4);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.5);
            border-radius: 10px;
        }
        
        /* 圖表容器 */
        .chart-container {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(15px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 20px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        }
        
        /* 隱藏 Streamlit 預設元素 */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* 響應式設計 */
        @media (max-width: 768px) {
            .main .block-container {
                padding: 1rem;
                margin: 0.5rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # 側邊欄 - 次要功能
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; padding: 1rem; background: rgba(255, 255, 255, 0.3); border-radius: 15px; margin-bottom: 2rem; backdrop-filter: blur(10px);'>
                <h2 style='margin: 0; color: #2c3e50;'>⚙️ Controls</h2>
                <p style='margin: 0.5rem 0 0 0; color: #7f8c8d; font-size: 0.9rem;'>Adjust parameters & settings</p>
            </div>
        """, unsafe_allow_html=True)
        
        # 預設參數組合
        st.markdown("### 🎯 Quick Presets")
        presets = {
            "Perfect Fit": {"a": 2.0, "b": 1.0, "noise": 0.5, "n_points": 100},
            "Steep Trend": {"a": 5.0, "b": 0.0, "noise": 2.0, "n_points": 150},
            "Gentle Slope": {"a": 0.5, "b": 3.0, "noise": 1.5, "n_points": 80},
            "Noisy Data": {"a": 1.5, "b": 2.0, "noise": 5.0, "n_points": 200}
        }
        
        # 使用 session_state 來存儲預設選擇
        if 'preset_params' not in st.session_state:
            st.session_state.preset_params = {"a": 2.0, "b": 1.0, "noise": 2.0, "n_points": 100}
        
        selected_preset = st.selectbox(
            "Choose a preset:",
            options=list(presets.keys()),
            index=0
        )
        
        if st.button("� Apply Preset", type="primary"):
            st.session_state.preset_params = presets[selected_preset]
            st.rerun()

        st.markdown("---")
        
        # 參數控制
        st.markdown("### 🎛️ Parameters")
        
        a = st.slider(
            '📈 Slope (a)', 
            min_value=-10.0, 
            max_value=10.0, 
            value=st.session_state.preset_params["a"], 
            step=0.1,
            help="Controls the steepness of the line"
        )
        
        b = st.slider(
            '📍 Intercept (b)', 
            min_value=-20.0, 
            max_value=20.0, 
            value=st.session_state.preset_params["b"], 
            step=0.1,
            help="Y-axis intercept value"
        )
        
        noise = st.slider(
            '🎲 Noise Level', 
            min_value=0.0, 
            max_value=10.0, 
            value=st.session_state.preset_params["noise"], 
            step=0.1,
            help="Standard deviation of random noise"
        )
        
        n_points = st.slider(
            '🔢 Data Points', 
            min_value=10, 
            max_value=500, 
            value=st.session_state.preset_params["n_points"], 
            step=10,
            help="Number of data points to generate"
        )
        
        random_state = st.number_input(
            '🌱 Random Seed', 
            value=42, 
            step=1,
            help="Seed for reproducible results"
        )

        st.markdown("---")
        
        # 其他控制選項
        st.markdown("### 🎨 Display Options")
        
        show_true_line = st.checkbox("Show True Line (No Noise)", value=True)
        show_confidence = st.checkbox("Show Confidence Interval", value=False)
        
        st.markdown("---")
        
        # 預測功能
        st.markdown("### 🔮 Prediction")
        predict_x = st.number_input('X value:', value=0.0, step=0.1)
        
        if st.button("🎯 Predict Y", type="primary"):
            st.session_state.prediction_requested = True
            st.session_state.predict_x = predict_x

        st.markdown("---")
        
        # 數據生成
        if st.button("🎲 Generate New Data", type="primary"):
            st.session_state.random_state = np.random.randint(1, 1000)
            st.rerun()

    # 主內容區域
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1 style='font-size: 3rem; margin-bottom: 0.5rem; color: #2c3e50;'>
                Linear Regression Demo
            </h1>
            <p style='font-size: 1.2rem; color: #7f8c8d; margin: 0;'>
                Interactive visualization with glassmorphism design
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # 側邊欄提示
    st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(52, 152, 219, 0.1), rgba(155, 89, 182, 0.1)); 
                    border: 2px solid rgba(52, 152, 219, 0.3); 
                    border-radius: 15px; 
                    padding: 1rem; 
                    margin-bottom: 2rem; 
                    text-align: center;
                    backdrop-filter: blur(10px);'>
            <h4 style='margin: 0 0 0.5rem 0; color: #3498db;'>
                🎛️ Use the sidebar to adjust parameters!
            </h4>
            <p style='margin: 0; color: #7f8c8d; font-size: 0.9rem;'>
                Click the <strong style='color: #3498db;'>blue button</strong> in the top-left corner to open/close the control panel
            </p>
        </div>
    """, unsafe_allow_html=True)

    # 生成數據和訓練模型
    current_random_state = getattr(st.session_state, 'random_state', random_state)
    df = generate_linear_data(a=a, b=b, noise=noise, n_points=n_points, random_state=int(current_random_state))
    X_train, X_test, y_train, y_test = train_test_split(df['x'], df['y'], test_size=0.2, random_state=int(random_state))
    model = train_linear_regression(X_train, y_train)
    y_pred = predict(model, X_test)
    y_train_pred = predict(model, X_train)
    metrics = evaluate(y_test, y_pred)

    st.markdown("<hr style='border: none; height: 2px; background: linear-gradient(45deg, #667eea, #764ba2); margin: 2em 0; border-radius: 2px;'/>", unsafe_allow_html=True)

    # 主要圖表區域
    # 主要可視化區域
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # 使用 Plotly 創建互動圖表
        fig = go.Figure()
        
        # 訓練數據點
        fig.add_trace(go.Scatter(
            x=X_train, y=y_train,
            mode='markers',
            name='🔵 Training Data',
            marker=dict(
                size=8,
                color='rgba(102, 126, 234, 0.7)',
                line=dict(width=2, color='rgba(102, 126, 234, 1)')
            ),
            hovertemplate='<b>Training Point</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        ))
        
        # 測試數據點
        fig.add_trace(go.Scatter(
            x=X_test, y=y_test,
            mode='markers',
            name='🟠 Test Data',
            marker=dict(
                size=8,
                color='rgba(255, 149, 0, 0.7)',
                line=dict(width=2, color='rgba(255, 149, 0, 1)')
            ),
            hovertemplate='<b>Test Point</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        ))
        
        # 回歸線
        x_line = np.linspace(df['x'].min()-1, df['x'].max()+1, 100)
        y_line = model.predict(x_line.reshape(-1, 1))
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode='lines',
            name='📈 Regression Line',
            line=dict(color='rgba(17, 17, 17, 0.8)', width=3),
            hovertemplate='<b>Predicted</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        ))
        
        # 理想線（無噪音）
        y_ideal = a * x_line + b
        fig.add_trace(go.Scatter(
            x=x_line, y=y_ideal,
            mode='lines',
            name='✨ True Line (No Noise)',
            line=dict(color='rgba(118, 75, 162, 0.6)', width=2, dash='dash'),
            hovertemplate='<b>True Line</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text=f"<b>Linear Regression: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}</b>",
                font=dict(size=18),
                x=0.5
            ),
            xaxis_title="X Values",
            yaxis_title="Y Values",
            hovermode='closest',
            template="plotly_white",
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # 模型指標
        st.markdown(f"""
            <div class="metric-glass">
                <h3 style='margin: 0; color: #2c3e50;'>MSE</h3>
                <h2 style='margin: 0.5rem 0; color: #e74c3c; font-size: 2rem;'>{metrics['mse']:.3f}</h2>
                <p style='margin: 0; color: #7f8c8d; font-size: 0.9rem;'>Mean Squared Error</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="metric-glass">
                <h3 style='margin: 0; color: #2c3e50;'>R² Score</h3>
                <h2 style='margin: 0.5rem 0; color: #27ae60; font-size: 2rem;'>{metrics['r2']:.3f}</h2>
                <p style='margin: 0; color: #7f8c8d; font-size: 0.9rem;'>Coefficient of Determination</p>
            </div>
        """, unsafe_allow_html=True)
        
        # 預測結果顯示
        if hasattr(st.session_state, 'prediction_requested') and st.session_state.prediction_requested:
            pred_y = model.predict([[st.session_state.predict_x]])[0]
            true_y = a * st.session_state.predict_x + b
            
            st.markdown(f"""
                <div class="metric-glass">
                    <h3 style='margin: 0; color: #2c3e50;'>Prediction</h3>
                    <p style='margin: 0.5rem 0 0 0; color: #3498db;'><strong>X:</strong> {st.session_state.predict_x:.2f}</p>
                    <p style='margin: 0; color: #9b59b6;'><strong>Predicted Y:</strong> {pred_y:.2f}</p>
                    <p style='margin: 0; color: #f39c12;'><strong>True Y:</strong> {true_y:.2f}</p>
                    <p style='margin: 0; color: #e74c3c;'><strong>Error:</strong> {abs(pred_y - true_y):.2f}</p>
                </div>
            """, unsafe_allow_html=True)
    
    # 殘差分析
    st.markdown("<h2 style='text-align:center; margin: 2rem 0 1rem 0;'>📉 Residual Analysis</h2>", unsafe_allow_html=True)
    
    residuals_train = y_train - y_train_pred
    residuals_test = y_test - y_pred
    
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        fig_residuals = go.Figure()
        
        fig_residuals.add_trace(go.Scatter(
            x=y_train_pred, y=residuals_train,
            mode='markers',
            name='Training Residuals',
            marker=dict(size=6, color='rgba(102, 126, 234, 0.6)'),
            hovertemplate='<b>Training</b><br>Predicted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>'
        ))
        
        fig_residuals.add_trace(go.Scatter(
            x=y_pred, y=residuals_test,
            mode='markers',
            name='Test Residuals',
            marker=dict(size=6, color='rgba(255, 149, 0, 0.6)'),
            hovertemplate='<b>Test</b><br>Predicted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>'
        ))
        
        # 零殘差線
        fig_residuals.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.7)
        
        fig_residuals.update_layout(
            title="Residuals vs Predicted Values",
            xaxis_title="Predicted Values",
            yaxis_title="Residuals",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig_residuals, use_container_width=True)
    
    with col_res2:
        # 殘差分布直方圖
        fig_hist = go.Figure()
        
        fig_hist.add_trace(go.Histogram(
            x=np.concatenate([residuals_train, residuals_test]),
            nbinsx=20,
            name='Residuals Distribution',
            marker_color='rgba(102, 126, 234, 0.7)',
            hovertemplate='<b>Residual Range</b><br>Count: %{y}<extra></extra>'
        ))
        
        fig_hist.update_layout(
            title="Residuals Distribution",
            xaxis_title="Residual Values",
            yaxis_title="Frequency",
            template="plotly_white",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)

    # 預測功能
    st.markdown("<h2 style='text-align:center; margin: 2rem 0 1rem 0;'>🔮 Make Predictions</h2>", unsafe_allow_html=True)
    
    col_pred1, col_pred2, col_pred3 = st.columns([1, 1, 1])
    
    with col_pred1:
        predict_x = st.number_input('Enter X value for prediction:', value=0.0, step=0.1)
    
    with col_pred2:
        if st.button("🎯 Predict", type="primary"):
            predicted_y = model.predict([[predict_x]])[0]
            true_y = a * predict_x + b
            st.success(f"**Predicted Y**: {predicted_y:.2f}")
            st.info(f"**True Y (no noise)**: {true_y:.2f}")
            st.warning(f"**Difference**: {abs(predicted_y - true_y):.2f}")
    
    with col_pred3:
        st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.9); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                <p style='margin: 0; color: #666; font-size: 0.9rem;'>
                    💡 Enter any X value to see what the model predicts vs the true value
                </p>
            </div>
        """, unsafe_allow_html=True)

# ====== 線性資料集產生器 ======
def generate_linear_data(a=1.0, b=0.0, noise=1.0, n_points=100, random_state=None):
    """
    產生線性資料集 y = ax + b + noise
    """
    rng = np.random.default_rng(random_state)
    x = rng.uniform(-10, 10, n_points)
    noise_arr = rng.normal(0, noise, n_points)
    y = a * x + b + noise_arr
    return pd.DataFrame({'x': x, 'y': y})

# ====== 線性迴歸模型訓練 ======
def train_linear_regression(X, y):
    X = np.array(X).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    return model

# ====== 預測 ======
def predict(model, X):
    X = np.array(X).reshape(-1, 1)
    return model.predict(X)

# ====== 評估 ======
def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'mse': mse, 'r2': r2}

if __name__ == '__main__':
    main()