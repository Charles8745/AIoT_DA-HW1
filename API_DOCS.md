# API Documentation

## 📋 Function Reference

This document provides detailed API documentation for all functions and components in the Linear Regression Studio application.

---

## 🔧 Core Functions

### `generate_linear_data(a=1.0, b=0.0, noise=1.0, n_points=100, random_state=None)`

Generates synthetic linear dataset following the equation `y = ax + b + noise`.

**Parameters:**
- `a` (float, default=1.0): Slope of the linear relationship
- `b` (float, default=0.0): Y-intercept of the linear relationship  
- `noise` (float, default=1.0): Standard deviation of Gaussian noise added to y values
- `n_points` (int, default=100): Number of data points to generate
- `random_state` (int, optional): Random seed for reproducible results

**Returns:**
- `pandas.DataFrame`: DataFrame with columns 'x' and 'y' containing generated data

**Example:**
```python
df = generate_linear_data(a=2.0, b=1.0, noise=0.5, n_points=150)
# Creates 150 points following y = 2x + 1 + noise
```

---

### `train_linear_regression(X, y)`

Trains a linear regression model using scikit-learn's LinearRegression.

**Parameters:**
- `X` (array-like): Input features (1D array will be reshaped to 2D)
- `y` (array-like): Target values

**Returns:**
- `sklearn.linear_model.LinearRegression`: Fitted linear regression model

**Example:**
```python
model = train_linear_regression(X_train, y_train)
```

---

### `predict(model, X)`

Makes predictions using a trained linear regression model.

**Parameters:**
- `model` (sklearn.linear_model.LinearRegression): Trained model
- `X` (array-like): Input features for prediction

**Returns:**
- `numpy.ndarray`: Predicted values

**Example:**
```python
predictions = predict(model, X_test)
```

---

### `evaluate(y_true, y_pred)`

Calculates evaluation metrics for regression model performance.

**Parameters:**
- `y_true` (array-like): True target values
- `y_pred` (array-like): Predicted target values

**Returns:**
- `dict`: Dictionary containing:
  - `'mse'` (float): Mean Squared Error
  - `'r2'` (float): R² Score (Coefficient of Determination)

**Example:**
```python
metrics = evaluate(y_test, y_pred)
print(f"MSE: {metrics['mse']:.3f}, R²: {metrics['r2']:.3f}")
```

---

## 🎨 UI Component Classes

### CSS Class Reference

#### `.glass-card`
Creates glassmorphism effect for main content containers.
```css
background: rgba(255, 255, 255, 0.15);
backdrop-filter: blur(20px);
border-radius: 20px;
border: 1px solid rgba(255, 255, 255, 0.2);
```

#### `.glass-title`
Glassmorphism styling for title sections.
```css
background: rgba(255, 255, 255, 0.1);
backdrop-filter: blur(15px);
border-radius: 20px;
```

#### `.metric-glass`
Glass effect for metric display cards.
```css
background: rgba(255, 255, 255, 0.15);
backdrop-filter: blur(20px);
border-radius: 15px;
```

---

## 📊 Chart Configuration

### Plotly Figure Settings

#### Main Scatter Plot Configuration
```python
fig = go.Figure()

# Training data points
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
```

#### Chart Layout Configuration
```python
fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    height=600,
    xaxis=dict(gridcolor='rgba(255, 255, 255, 0.2)'),
    yaxis=dict(gridcolor='rgba(255, 255, 255, 0.2)')
)
```

---

## 🎛️ Streamlit Component API

### Session State Variables

#### `st.session_state.preset_params`
Stores current parameter configuration.

**Structure:**
```python
{
    'a': float,      # Slope value
    'b': float,      # Intercept value  
    'noise': float,  # Noise level
    'n_points': int  # Number of data points
}
```

### Sidebar Components

#### Parameter Sliders
```python
a = st.slider('Slope (a)', min_value=-10.0, max_value=10.0, 
             value=st.session_state.preset_params["a"], step=0.1)
```

#### Preset Selection
```python
preset_choice = st.selectbox(
    "Choose Preset:", 
    ["Custom"] + list(presets.keys())
)
```

#### Conditional Rendering
```python
show_residuals = st.checkbox("Show Residuals", value=True)
show_true_line = st.checkbox("Show True Line", value=True)
```

---

## 🔍 Data Structures

### Preset Configurations
```python
presets = {
    "Perfect Fit": {"a": 2.0, "b": 1.0, "noise": 0.5, "n_points": 100},
    "Steep Trend": {"a": 5.0, "b": 0.0, "noise": 2.0, "n_points": 150},
    "Gentle Slope": {"a": 0.5, "b": 3.0, "noise": 1.5, "n_points": 80},
    "Noisy Data": {"a": 1.5, "b": 2.0, "noise": 5.0, "n_points": 200}
}
```

### Model Output Structure
```python
# Generated dataset
df: pd.DataFrame
├── 'x': float64  # Independent variable values
└── 'y': float64  # Dependent variable values (with noise)

# Train/test split
X_train, X_test: np.ndarray  # Feature arrays
y_train, y_test: np.ndarray  # Target arrays

# Model evaluation
metrics: dict
├── 'mse': float   # Mean Squared Error
└── 'r2': float    # R² Score
```

---

## 🎨 Animation & Effects

### CSS Animations

#### Gradient Background Animation
```css
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
```

#### Button Hover Effects
```css
.stButton > button:hover {
    background: rgba(255, 255, 255, 0.3);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}
```

---

## 🔧 Configuration Options

### Streamlit Configuration
```python
st.set_page_config(
    layout="wide", 
    page_title="Linear Regression Interactive Demo",
    page_icon="📊",
    initial_sidebar_state="expanded"
)
```

### Chart Responsiveness
```python
st.plotly_chart(fig, use_container_width=True)
```

---

## 📱 Responsive Design Breakpoints

### Desktop Optimization
```css
@media (min-width: 1200px) {
    .main .block-container {
        padding: 1rem 3rem;
    }
}
```

### Component Scaling
- **Sidebar width**: Auto-adjusting based on content
- **Main content**: 3:1 ratio (chart:metrics)
- **Chart height**: Fixed 600px for consistent viewing
- **Metric cards**: Flexible width with consistent padding

---

## 🚀 Performance Considerations

### Rendering Optimization
- **Conditional rendering**: Components only render when needed
- **Efficient updates**: Plotly charts update data without full re-render
- **State persistence**: Session state reduces unnecessary recalculations

### Memory Management
- **Data size limits**: Configurable maximum data points (500)
- **Chart complexity**: Optimized trace count for smooth interaction
- **CSS efficiency**: Minimal DOM manipulation through Streamlit

---

*This API documentation provides comprehensive reference for developers working with or extending the Linear Regression Studio application.*