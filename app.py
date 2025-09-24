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
    # Glassmorphism 風格配置
    st.set_page_config(
        layout="wide", 
        page_title="Linear Regression Interactive Demo",
        page_icon="📊",
        initial_sidebar_state="expanded"
    )
    
    # Glassmorphism CSS 樣式
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* 背景設定 */
        html, body, [class*="css"] {
            font-family: 'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
        }
        
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* 主容器 - 完全透明的玻璃效果 */
        .main .block-container {
            padding: 1rem 2rem;
            max-width: none;
            background: transparent;
            backdrop-filter: none;
        }
        
        /* Glassmorphism 卡片 */
        .glass-card {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }
        
        /* 側邊欄玻璃效果 */
        .stSidebar > div {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(15px);
            border-right: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .stSidebar .stSlider > div {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        /* 按鈕玻璃效果 */
        .stButton > button {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            color: white;
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        .stButton > button:hover {
            background: rgba(255, 255, 255, 0.3);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        }
        
        /* 標題玻璃效果 */
        .glass-title {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 2rem;
            text-align: center;
            margin: 1rem 0;
            color: white;
        }
        
        /* 指標卡片玻璃效果 */
        .metric-glass {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(20px);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 1.5rem;
            margin: 1rem 0;
            text-align: center;
            color: white;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
        }
        
        /* 輸入框玻璃效果 */
        .stNumberInput > div > div > input {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            color: white;
        }
        
        /* 文字顏色 */
        h1, h2, h3, h4, h5, h6 {
            color: white !important;
            font-weight: 600;
        }
        
        .stMarkdown, .stText, p {
            color: white !important;
        }
        
        /* 隱藏 Streamlit 默認元素 */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* 圖表背景透明化 */
        .stPlotlyChart {
            background: transparent;
        }
        
        /* 響應式設計 - 橫式優化 */
        @media (min-width: 1200px) {
            .main .block-container {
                padding: 1rem 3rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # 主標題 - 玻璃效果
    st.markdown("""
        <div class="glass-title">
            <h1 style='font-size: 2.8rem; margin: 0; font-weight: 700;'>
                📊 Linear Regression Studio
            </h1>
            <p style='font-size: 1.2rem; margin: 0.5rem 0 0 0; opacity: 0.9;'>
                Interactive Machine Learning Visualization
            </p>
        </div>
    """, unsafe_allow_html=True)

    # 側邊欄控制面板
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        
        # 預設參數組合
        st.markdown("#### 🚀 Quick Presets")
        presets = {
            "Perfect Fit": {"a": 2.0, "b": 1.0, "noise": 0.5, "n_points": 100},
            "Steep Trend": {"a": 5.0, "b": 0.0, "noise": 2.0, "n_points": 150},
            "Gentle Slope": {"a": 0.5, "b": 3.0, "noise": 1.5, "n_points": 80},
            "Noisy Data": {"a": 1.5, "b": 2.0, "noise": 5.0, "n_points": 200}
        }
        
        # 使用 session_state 來存儲預設選擇
        if 'preset_params' not in st.session_state:
            st.session_state.preset_params = {"a": 2.0, "b": 1.0, "noise": 2.0, "n_points": 100}
        
        preset_choice = st.selectbox(
            "Choose Preset:", 
            ["Custom"] + list(presets.keys())
        )
        
        if preset_choice != "Custom":
            st.session_state.preset_params = presets[preset_choice]
        
        st.markdown("---")
        
        # 參數控制
        st.markdown("#### 📈 Parameters")
        a = st.slider('Slope (a)', min_value=-10.0, max_value=10.0, 
                     value=st.session_state.preset_params["a"], step=0.1)
        b = st.slider('Intercept (b)', min_value=-20.0, max_value=20.0, 
                     value=st.session_state.preset_params["b"], step=0.1)
        noise = st.slider('Noise Level', min_value=0.0, max_value=10.0, 
                         value=st.session_state.preset_params["noise"], step=0.1)
        n_points = st.slider('Data Points', min_value=10, max_value=500, 
                            value=st.session_state.preset_params["n_points"], step=10)
        
        st.markdown("---")
        
        # 次要功能
        st.markdown("#### ⚙️ Advanced Settings")
        random_state = st.number_input('Random Seed', value=42, step=1)
        
        if st.button("🎲 Generate New Data", type="primary"):
            random_state = np.random.randint(1, 1000)
        
        show_residuals = st.checkbox("Show Residuals", value=True)
        show_true_line = st.checkbox("Show True Line", value=True)
        
        st.markdown("---")
        
        # 預測功能
        st.markdown("#### 🔮 Prediction")
        predict_x = st.number_input('X value:', value=0.0, step=0.1)
    
    # 生成數據和訓練模型
    df = generate_linear_data(a=a, b=b, noise=noise, n_points=n_points, random_state=int(random_state))
    X_train, X_test, y_train, y_test = train_test_split(df['x'], df['y'], test_size=0.2, random_state=int(random_state))
    model = train_linear_regression(X_train, y_train)
    y_pred = predict(model, X_test)
    y_train_pred = predict(model, X_train)
    metrics = evaluate(y_test, y_pred)
    
    # 預測結果
    if predict_x is not None:
        predicted_y = model.predict([[predict_x]])[0]
        true_y = a * predict_x + b
        with st.sidebar:
            st.success(f"**Predicted**: {predicted_y:.2f}")
            st.info(f"**True Value**: {true_y:.2f}")
            st.warning(f"**Error**: {abs(predicted_y - true_y):.2f}")

    # 主要內容區域 - 橫式布局
    col_main, col_metrics = st.columns([3, 1])
    
    with col_main:
        # 主圖表 - 玻璃容器
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # 使用 Plotly 創建互動圖表
        fig = go.Figure()
        
        # 訓練數據點
        fig.add_trace(go.Scatter(
            x=X_train, y=y_train,
            mode='markers',
            name='🔵 Training',
            marker=dict(
                size=8,
                color='rgba(102, 126, 234, 0.8)',
                line=dict(width=2, color='rgba(255, 255, 255, 0.6)')
            ),
            hovertemplate='<b>Training Point</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        ))
        
        # 測試數據點
        fig.add_trace(go.Scatter(
            x=X_test, y=y_test,
            mode='markers',
            name='🟠 Test',
            marker=dict(
                size=8,
                color='rgba(255, 149, 0, 0.8)',
                line=dict(width=2, color='rgba(255, 255, 255, 0.6)')
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
            line=dict(color='rgba(255, 255, 255, 0.9)', width=3),
            hovertemplate='<b>Predicted</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        ))
        
        # 理想線（可選）
        if show_true_line:
            y_ideal = a * x_line + b
            fig.add_trace(go.Scatter(
                x=x_line, y=y_ideal,
                mode='lines',
                name='✨ True Line',
                line=dict(color='rgba(255, 255, 255, 0.5)', width=2, dash='dash'),
                hovertemplate='<b>True Line</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
            ))
        
        # 預測點
        if predict_x is not None:
            fig.add_trace(go.Scatter(
                x=[predict_x], y=[predicted_y],
                mode='markers',
                name='🎯 Prediction',
                marker=dict(
                    size=12,
                    color='rgba(255, 255, 255, 0.9)',
                    symbol='star',
                    line=dict(width=2, color='rgba(255, 0, 0, 0.8)')
                ),
                hovertemplate='<b>Prediction</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
            ))
        
        # 透明背景設定
        fig.update_layout(
            title=dict(
                text=f"<b>y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}</b>",
                font=dict(size=20, color='white'),
                x=0.5
            ),
            xaxis_title="X Values",
            yaxis_title="Y Values",
            hovermode='closest',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(255, 255, 255, 0.1)',
                bordercolor='rgba(255, 255, 255, 0.2)',
                borderwidth=1
            ),
            xaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.2)',
                zerolinecolor='rgba(255, 255, 255, 0.3)'
            ),
            yaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.2)',
                zerolinecolor='rgba(255, 255, 255, 0.3)'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_metrics:
        # 指標顯示 - 玻璃效果
        st.markdown(f"""
            <div class="metric-glass">
                <h3 style='margin: 0;'>MSE</h3>
                <h2 style='margin: 0.5rem 0 0 0;'>{metrics['mse']:.3f}</h2>
                <p style='margin: 0; opacity: 0.8; font-size: 0.9rem;'>Mean Squared Error</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="metric-glass">
                <h3 style='margin: 0;'>R² Score</h3>
                <h2 style='margin: 0.5rem 0 0 0;'>{metrics['r2']:.3f}</h2>
                <p style='margin: 0; opacity: 0.8; font-size: 0.9rem;'>Coefficient of Determination</p>
            </div>
        """, unsafe_allow_html=True)
        
        # 數據統計
        st.markdown(f"""
            <div class="metric-glass">
                <h4 style='margin: 0;'>📊 Dataset Info</h4>
                <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>
                    Training: {len(X_train)} points<br>
                    Test: {len(X_test)} points<br>
                    Total: {n_points} points
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # 殘差分析（可選）
    if show_residuals:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📉 Residual Analysis")
        
        residuals_train = y_train - y_train_pred
        residuals_test = y_test - y_pred
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            fig_residuals = go.Figure()
            
            fig_residuals.add_trace(go.Scatter(
                x=y_train_pred, y=residuals_train,
                mode='markers',
                name='Training Residuals',
                marker=dict(size=6, color='rgba(102, 126, 234, 0.7)'),
                hovertemplate='<b>Training</b><br>Predicted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>'
            ))
            
            fig_residuals.add_trace(go.Scatter(
                x=y_pred, y=residuals_test,
                mode='markers',
                name='Test Residuals',
                marker=dict(size=6, color='rgba(255, 149, 0, 0.7)'),
                hovertemplate='<b>Test</b><br>Predicted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>'
            ))
            
            fig_residuals.add_hline(y=0, line_dash="dash", line_color="rgba(255, 255, 255, 0.6)", opacity=0.7)
            
            fig_residuals.update_layout(
                title="Residuals vs Predicted",
                xaxis_title="Predicted Values",
                yaxis_title="Residuals",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400,
                xaxis=dict(gridcolor='rgba(255, 255, 255, 0.2)'),
                yaxis=dict(gridcolor='rgba(255, 255, 255, 0.2)')
            )
            
            st.plotly_chart(fig_residuals, use_container_width=True)
        
        with col_res2:
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
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400,
                showlegend=False,
                xaxis=dict(gridcolor='rgba(255, 255, 255, 0.2)'),
                yaxis=dict(gridcolor='rgba(255, 255, 255, 0.2)')
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

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