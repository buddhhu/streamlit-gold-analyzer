# 🪙 Gold Price Analysis & Prediction

![Gold Bars](https://cms-resources.groww.in/uploads/Blog_Cover_Image_11_4a6a6bfd46.jpg)

## 📊 Overview

**Gold Price Analysis & Prediction** is a data-driven web application built with **Streamlit** that helps traders, investors, and data enthusiasts gain insights into historical gold price trends and forecast future prices. The app uses **ARIMA (AutoRegressive Integrated Moving Average)** time series modeling for predictions based on historical data.

## ✨ Features

* 🔄 **Live Data Scraping**: Fetch the latest gold price data from Groww\.in
* 📉 **Historical Trend Analysis**: Visualize gold price trends over decades
* 🔮 **ARIMA-based Predictions**: Forecast future gold prices with customizable parameters
* 📈 **Interactive Visualizations**: Explore data through multiple interactive charts
* 💰 **Investment Calculator**: Calculate potential ROI for gold investments
* ⌛ **Flexible Time Windows**: Analyze recent trends or full historical datasets

## 🚀 Getting Started

### ⚙️ Prerequisites

* 🐍 Python 3.7+
* 📦 pip

### 📥 Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/buddhhu/gold-price-analysis.git
   cd gold-price-analysis
   ```

2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

### 📚 Dependencies

This app requires the following Python packages:

* `streamlit`
* `pandas`
* `numpy`
* `plotly`
* `requests`
* `beautifulsoup4`
* `statsmodels`

Use the `requirements.txt` for quick installation.

## 🖥️ Usage

1. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. Open your browser and go to `http://localhost:8501`

3. Use the sidebar to:

   * ⚡ Fetch latest gold price or load sample data
   * 🛠️ Configure ARIMA parameters
   * ⏳ Set prediction timeframe
   * 🔍 View and explore raw data

4. Explore visualizations via tabs:

   * 📜 Full History & Predictions
   * 📊 Recent Trends
   * 📈 Statistics

5. Use the 💸 **ROI calculator** to estimate returns on gold investments

## 📊 Data Sources

Data is scraped from:

* 📜 [Historical Data](https://groww.in/blog/historical-gold-rates-trend-in-india)
* 💰 [Current Price](https://groww.in/gold-rates)

Or, use built-in sample data for demo purposes.

## 🧮 Technical Details

### ⚙️ ARIMA Model

The app uses ARIMA for time series forecasting:

* `p`: AR (AutoRegressive) — influence of past values
* `d`: Differencing — ensures stationarity
* `q`: MA (Moving Average) — influence of past errors

These can be tweaked via sidebar sliders.

### 📈 Visualizations

The app provides:

* 📉 Historical prices + future predictions
* 📊 Recent trends with forecast
* 📆 Yearly % change
* 💸 ROI estimation charts

## 🛠️ Customization

Easily modify the app by:

* Changing ARIMA parameters
* Updating prediction duration
* Adjusting historical time window
* Setting custom investment values

## 📄 License

Licensed under the **MIT License** — see the [LICENSE](LICENSE) for details.

## 👨‍💻 Author

Made with ❤️ by **Amit Sharma**

## 🙏 Acknowledgements

* 📊 Data Source: [Groww.in](https://groww.in)
* 🧰 Built using: [Streamlit](https://streamlit.io/)
* 📈 Forecasting: [statsmodels](https://www.statsmodels.org/)
* 🎨 Visuals: [Plotly](https://plotly.com/)