import streamlit as st

def main():
	st.title('簡單線性迴歸互動展示')
	st.markdown('---')
	st.header('參數設定')
	a = st.slider('斜率 a', min_value=-10.0, max_value=10.0, value=2.0, step=0.1)
	b = st.slider('截距 b', min_value=-20.0, max_value=20.0, value=1.0, step=0.1)
	noise = st.slider('雜訊標準差', min_value=0.0, max_value=10.0, value=2.0, step=0.1)
	n_points = st.slider('資料點數', min_value=10, max_value=500, value=100, step=1)
	random_state = st.number_input('隨機種子', value=42, step=1)

	df = generate_linear_data(a=a, b=b, noise=noise, n_points=n_points, random_state=int(random_state))
	X_train, X_test, y_train, y_test = train_test_split(df['x'], df['y'], test_size=0.2, random_state=int(random_state))
	model = train_linear_regression(X_train, y_train)
	y_pred = predict(model, X_test)
	metrics = evaluate(y_test, y_pred)

	st.header('資料與回歸線')
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(figsize=(8, 5))
	ax.scatter(X_train, y_train, color='blue', label='Train')
	ax.scatter(X_test, y_test, color='orange', label='Test')
	x_line = np.linspace(df['x'].min(), df['x'].max(), 100)
	y_line = model.predict(x_line.reshape(-1, 1))
	ax.plot(x_line, y_line, color='red', label='Regression Line')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_title('資料與回歸線')
	ax.legend()
	st.pyplot(fig)

	st.header('模型評估指標')
	st.write(f"MSE: {metrics['mse']:.3f}")
	st.write(f"R2: {metrics['r2']:.3f}")

if __name__ == '__main__':
	main()
import matplotlib.pyplot as plt

# 資料與模型視覺化
def plot_regression(X, y, model=None, title='Linear Regression', show=True):
	"""
	繪製資料點與回歸線
	X: 特徵（1D array-like）
	y: 標籤
	model: 已訓練的線性迴歸模型（可選）
	title: 圖表標題
	show: 是否直接顯示
	"""
	X = np.array(X)
	y = np.array(y)
	plt.figure(figsize=(8, 5))
	plt.scatter(X, y, color='blue', label='Data')
	if model is not None:
		x_line = np.linspace(X.min(), X.max(), 100)
		y_line = model.predict(x_line.reshape(-1, 1))
		plt.plot(x_line, y_line, color='red', label='Regression Line')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title(title)
	plt.legend()
	if show:
		plt.show()
	else:
		return plt
from sklearn.model_selection import train_test_split

# CRISP-DM 流程範例：資料分割與建模
def crispdm_example(a=2.0, b=1.0, noise=2.0, n_points=100, test_size=0.2, random_state=42):
	# 1. 產生資料
	df = generate_linear_data(a=a, b=b, noise=noise, n_points=n_points, random_state=random_state)
	# 2. 分割訓練/測試集
	X_train, X_test, y_train, y_test = train_test_split(df['x'], df['y'], test_size=test_size, random_state=random_state)
	# 3. 訓練模型
	model = train_linear_regression(X_train, y_train)
	# 4. 預測
	y_pred = predict(model, X_test)
	# 5. 評估
	metrics = evaluate(y_test, y_pred)
	return {
		'model': model,
		'metrics': metrics,
		'X_test': X_test,
		'y_test': y_test,
		'y_pred': y_pred,
		'params': {'a': a, 'b': b, 'noise': noise, 'n_points': n_points, 'test_size': test_size}
	}
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 線性迴歸模型訓練
def train_linear_regression(X, y):
	"""
	訓練簡單線性迴歸模型
	X: 特徵（1D 或 2D array-like）
	y: 標籤
	回傳：已訓練的模型
	"""
	X = np.array(X).reshape(-1, 1)
	model = LinearRegression()
	model.fit(X, y)
	return model

# 預測
def predict(model, X):
	X = np.array(X).reshape(-1, 1)
	return model.predict(X)

# 評估
def evaluate(y_true, y_pred):
	mse = mean_squared_error(y_true, y_pred)
	r2 = r2_score(y_true, y_pred)
	return {'mse': mse, 'r2': r2}
# 線性資料集產生器
import numpy as np
import pandas as pd

def generate_linear_data(a=1.0, b=0.0, noise=1.0, n_points=100, random_state=None):
	"""
	產生線性資料集 y = ax + b + noise
	參數：
		a: 斜率
		b: 截距
		noise: 雜訊標準差
		n_points: 資料點數量
		random_state: 隨機種子
	回傳：
		DataFrame，包含 x, y 欄位
	"""
	rng = np.random.default_rng(random_state)
	x = rng.uniform(-10, 10, n_points)
	noise_arr = rng.normal(0, noise, n_points)
	y = a * x + b + noise_arr
	return pd.DataFrame({'x': x, 'y': y})
