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
        initial_sidebar_state="collapsed"
    )
    
    # 更現代化的 CSS 樣式
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            margin: 2rem auto;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        .stSlider > div { 
            padding: 0.5rem 0;
            background: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            margin: 0.5rem 0;
            padding: 1rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .stButton > button {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border-radius: 25px;
            border: none;
            padding: 0.75rem 2rem;
            font-size: 1.1rem;
            font-weight: 600;
            margin: 1rem 0.5rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        .metric-card {
            background: linear-gradient(135deg, #6DD5ED, #2193b0);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin: 1rem 0;
            box-shadow: 0 10px 25px rgba(33, 147, 176, 0.2);
        }
        h1, h2, h3 {
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        .stNumberInput > div > div > input {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            border: 2px solid rgba(102, 126, 234, 0.2);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # 主標題
    st.markdown("""
        <h1 style='font-size:3.2rem; margin-bottom:0.5em; text-align:center;'>
            Simple Linear Regression Interactive Demo
        </h1>
        <div style='text-align:center; margin-bottom:2em;'>
            <span style='background: linear-gradient(45deg, #667eea, #764ba2); color: white; padding: 0.5rem 1.5rem; border-radius: 25px; font-size: 1.1rem; font-weight: 500;'>
                🎯 Powered by AI & Modern Design
            </span>
        </div>
        <hr style='border: none; height: 2px; background: linear-gradient(45deg, #667eea, #764ba2); margin: 2em 0; border-radius: 2px;'/>
    """, unsafe_allow_html=True)

    # 預設參數組合
    st.markdown("<h2 style='text-align:center; margin: 1rem 0;'>🚀 Quick Presets</h2>", unsafe_allow_html=True)
    col_preset1, col_preset2, col_preset3, col_preset4 = st.columns(4)
    
    presets = {
        "Perfect Fit": {"a": 2.0, "b": 1.0, "noise": 0.5, "n_points": 100},
        "Steep Trend": {"a": 5.0, "b": 0.0, "noise": 2.0, "n_points": 150},
        "Gentle Slope": {"a": 0.5, "b": 3.0, "noise": 1.5, "n_points": 80},
        "Noisy Data": {"a": 1.5, "b": 2.0, "noise": 5.0, "n_points": 200}
    }
    
    # 使用 session_state 來存儲預設選擇
    if 'preset_params' not in st.session_state:
        st.session_state.preset_params = {"a": 2.0, "b": 1.0, "noise": 2.0, "n_points": 100}
    
    with col_preset1:
        if st.button("🎯 Perfect Fit"):
            st.session_state.preset_params = presets["Perfect Fit"]
    with col_preset2:
        if st.button("📈 Steep Trend"):
            st.session_state.preset_params = presets["Steep Trend"]
    with col_preset3:
        if st.button("📉 Gentle Slope"):
            st.session_state.preset_params = presets["Gentle Slope"]
    with col_preset4:
        if st.button("🌪️ Noisy Data"):
            st.session_state.preset_params = presets["Noisy Data"]

    # 參數控制面板
    st.markdown("<h2 style='text-align:center; margin: 2rem 0 1rem 0;'>🎛️ Parameters Control Panel</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    col4, col5, col6 = st.columns([1, 1, 1])
    
    with col1:
        a = st.slider('📈 Slope (a)', min_value=-10.0, max_value=10.0, 
                     value=st.session_state.preset_params["a"], step=0.1)
    with col2:
        b = st.slider('📍 Intercept (b)', min_value=-20.0, max_value=20.0, 
                     value=st.session_state.preset_params["b"], step=0.1)
    with col3:
        noise = st.slider('🎲 Noise Level', min_value=0.0, max_value=10.0, 
                         value=st.session_state.preset_params["noise"], step=0.1)
    with col4:
        n_points = st.slider('🔢 Data Points', min_value=10, max_value=500, 
                            value=st.session_state.preset_params["n_points"], step=10)
    with col5:
        random_state = st.number_input('🌱 Random Seed', value=42, step=1)
    with col6:
        regenerate = st.button("🎲 Generate New Data", type="primary")

    # 生成數據和訓練模型
    if regenerate:
        random_state = np.random.randint(1, 1000)
    
    df = generate_linear_data(a=a, b=b, noise=noise, n_points=n_points, random_state=int(random_state))
    X_train, X_test, y_train, y_test = train_test_split(df['x'], df['y'], test_size=0.2, random_state=int(random_state))
    model = train_linear_regression(X_train, y_train)
    y_pred = predict(model, X_test)
    y_train_pred = predict(model, X_train)
    metrics = evaluate(y_test, y_pred)

    st.markdown("<hr style='border: none; height: 2px; background: linear-gradient(45deg, #667eea, #764ba2); margin: 2em 0; border-radius: 2px;'/>", unsafe_allow_html=True)

    # 主要圖表區域
    col_chart1, col_chart2 = st.columns([2, 1])
    
    with col_chart1:
        st.markdown("<h2 style='text-align:center;'>📊 Interactive Visualization</h2>", unsafe_allow_html=True)
        
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

    with col_chart2:
        st.markdown("<h3 style='text-align:center; margin-bottom: 2rem;'>📈 Model Metrics</h3>", unsafe_allow_html=True)
        
        # 美觀的指標卡片
        st.markdown(f"""
            <div class="metric-card">
                <h3 style='margin: 0; color: white;'>MSE</h3>
                <h2 style='margin: 0.5rem 0 0 0; color: white;'>{metrics['mse']:.3f}</h2>
                <p style='margin: 0; opacity: 0.8;'>Mean Squared Error</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #FF6B6B, #FF8E53);">
                <h3 style='margin: 0; color: white;'>R² Score</h3>
                <h2 style='margin: 0.5rem 0 0 0; color: white;'>{metrics['r2']:.3f}</h2>
                <p style='margin: 0; opacity: 0.8;'>Coefficient of Determination</p>
            </div>
        """, unsafe_allow_html=True)
        
        # 數據統計
        st.markdown(f"""
            <div style="background: rgba(255, 255, 255, 0.9); padding: 1.5rem; border-radius: 15px; margin: 1rem 0; border-left: 4px solid #667eea;">
                <h4 style='margin: 0; color: #333;'>📊 Data Statistics</h4>
                <p style='margin: 0.5rem 0 0 0; color: #555;'>
                    Training Points: {len(X_train)}<br>
                    Test Points: {len(X_test)}<br>
                    Total Points: {n_points}
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # 回歸方程式
        st.markdown(f"""
            <div style="background: rgba(255, 255, 255, 0.9); padding: 1.5rem; border-radius: 15px; margin: 1rem 0; text-align: center; border-left: 4px solid #667eea;">
                <h4 style='margin: 0; color: #333;'>📐 Regression Equation</h4>
                <p style='font-size: 1.2rem; font-weight: 600; margin: 0.5rem 0 0 0; color: #667eea;'>
                    y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}
                </p>
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