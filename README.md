# Simple Linear Regression Interactive Demo

An interactive web application built with Python and Streamlit that demonstrates simple linear regression following the CRISP-DM methodology. Users can adjust parameters in real-time and observe how they affect the data distribution, regression line, and model performance.

## 🚀 Live Demo
Visit the live demo: [https://aiotda-hw1-y7kjqht7iguufuabbq8jbp.streamlit.app/](https://aiotda-hw1-y7kjqht7iguufuabbq8jbp.streamlit.app/)

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Cloud Deployment](#cloud-deployment)
- [Parameters Explained](#parameters-explained)
- [CRISP-DM Documentation](#crisp-dm-documentation)
- [Troubleshooting](#troubleshooting)

## Project Overview
This project implements an interactive simple linear regression demonstration using Python and Streamlit. The application follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology and features:

- **Clean, Modern Interface**: Inspired by Apple's design principles with minimal, desktop-friendly UI
- **Real-time Parameter Adjustment**: Interactive sliders for slope, intercept, noise, and data points
- **Instant Visualization**: Live updates of data scatter plots and regression lines
- **Model Performance Metrics**: Real-time MSE and R² score calculations

## Features
✨ **Interactive Data Generation**: Create synthetic linear datasets with customizable parameters  
📊 **Real-time Visualization**: Instantly see how parameter changes affect the data and model  
🎯 **Model Training & Evaluation**: Automatic linear regression training with performance metrics  
🌐 **Web-based Interface**: Streamlit-powered UI with modern, Apple-inspired design  
📈 **CRISP-DM Workflow**: Complete data science methodology documentation  

## Installation

### Prerequisites
- Python 3.8 or higher (Python 3.10 recommended)
- Git (for cloning the repository)

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Charles8745/AIoT_DA-HW1.git
   cd AIoT_DA-HW1
   ```

2. **Create Virtual Environment (Recommended)**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Required Packages**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually if requirements.txt is missing:
   ```bash
   pip install numpy pandas scikit-learn matplotlib streamlit
   ```

## Usage

### Local Development
1. **Navigate to Project Directory**
   ```bash
   cd AIoT_DA-HW1
   ```

2. **Activate Virtual Environment** (if created)
   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```

4. **Open in Browser**
   - The app will automatically open in your default browser
   - Or manually visit: http://localhost:8501

## Cloud Deployment

Deploy your app for free on Streamlit Community Cloud to share with anyone:

### Quick Deployment Steps

1. **Push to GitHub**
   - Ensure your project is in a GitHub repository
   - All files (app.py, requirements.txt) should be committed

2. **Deploy on Streamlit Cloud**
   - Visit [Streamlit Community Cloud](https://streamlit.io/cloud)
   - Sign in with your GitHub account
   - Click "New app" and select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Get Your Public URL**
   - After deployment, you'll receive a public URL
   - Share this URL with anyone to use your app

> 📖 **Detailed Guide**: [Official Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app)

## Parameters Explained

| Parameter | Description | Range | Effect |
|-----------|-------------|-------|--------|
| **Slope (a)** | Controls the steepness of the linear relationship | -10.0 to 10.0 | Higher values create steeper lines |
| **Intercept (b)** | Y-axis intercept of the regression line | -20.0 to 20.0 | Shifts the line up or down |
| **Noise Std** | Standard deviation of random noise added to data | 0.0 to 10.0 | Higher values scatter data more |
| **Data Points** | Number of data points to generate | 10 to 500 | More points give smoother visualizations |
| **Random Seed** | Seed for reproducible random data generation | Any integer | Same seed = same data pattern |

## CRISP-DM Documentation

This project follows the CRISP-DM methodology for data science projects:

- **📋 Development Log**: See `log.md` for detailed development progress and decisions
- **📋 Project Planning**: See `project_plan_AImodify.md` for project objectives and AI-assisted development workflow
- **📋 Manual Planning**: See `project_plan_manual.md` for manual development approach

## Troubleshooting

### Common Issues and Solutions

**🔧 Package Installation Fails**
- Ensure Python 3.8+ is installed: `python --version`
- Try upgrading pip: `pip install --upgrade pip`
- Check internet connection

**🔧 Streamlit Won't Start**
- Verify all packages are installed: `pip list`
- Try reinstalling streamlit: `pip install --force-reinstall streamlit`
- Check if port 8501 is available

**🔧 App Shows Errors on Streamlit Cloud**
- Verify requirements.txt contains all dependencies
- Check that main file is named `app.py`
- Ensure repository is public or properly configured

**🔧 Need Help?**
- Create an issue on GitHub
- Check the development log in `log.md`
- Review the troubleshooting section above

---

## 📫 Contributing & Support

This project is open for educational and research purposes. Feel free to:
- 🍴 Fork the repository
- 🐛 Report issues
- 💡 Suggest improvements
- 📚 Use for learning and teaching

**Made with ❤️ using Python and Streamlit**
