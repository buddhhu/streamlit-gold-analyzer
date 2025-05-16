import datetime
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup as bs
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Gold Price Analysis & Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FFD700;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #B8860B;
        margin-bottom: 1rem;
    }
    .sidebar .sidebar-content {
        background-color: #f5f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <h1 class="main-header" style="text-align: center;">📈 Gold Price Analysis & Prediction</h1>
    <p style="text-align: center; font-size: 1.1rem; color: #555;">
        Explore historical gold price trends, visualize fluctuations over time, and forecast upcoming prices using ARIMA-based time series modeling.
        This tool is built for traders, investors, and data enthusiasts who want data-backed insights into the gold market.
    </p>
""",
    unsafe_allow_html=True,
)


st.sidebar.image(
    "https://cms-resources.groww.in/uploads/Blog_Cover_Image_11_4a6a6bfd46.jpg",
    width=100,
)
st.sidebar.header("Controls")


def scrape_historical_data():
    st.sidebar.markdown("### Data Collection")

    if st.sidebar.button("Fetch Latest Data"):
        with st.spinner("Scraping historical gold price data..."):
            try:
                historical_url = (
                    "https://groww.in/blog/historical-gold-rates-trend-in-india"
                )
                response = requests.get(historical_url)
                soup = bs(response.text, "html.parser")

                table = soup.find("table")
                table_cells = soup.find_all("td")[3:]

                cell_values = [
                    cell.get_text(strip=True).replace("₹", "").replace(",", "")
                    for cell in table_cells
                ]

                gold_price_data = []
                for i in range(0, len(cell_values), 2):
                    try:
                        year = int(cell_values[i][:4])
                        price = float(cell_values[i + 1])
                        gold_price_data.append([year, price])
                    except (ValueError, IndexError):
                        continue

                st.sidebar.success(
                    f"✅ Retrieved {len(gold_price_data)} years of historical data"
                )

                with st.spinner("Getting latest gold price..."):
                    current_url = "https://groww.in/gold-rates"
                    response = requests.get(current_url)
                    soup = bs(response.text, "html.parser")

                    price_container = soup.find(
                        "div", {"class": "absolute-center bodyLargeHeavy"}
                    )
                    price_element = price_container.find("span", {"class": "bodyLarge"})
                    current_price = float(price_element.text.strip())
                    current_year = datetime.datetime.now().year

                    latest_data = [current_year, current_price]
                    st.sidebar.success(
                        f"✅ Latest gold price: ₹{current_price} ({current_year})"
                    )

                combined_data = gold_price_data + [latest_data]
                gold_df = pd.DataFrame(combined_data, columns=["Year", "Price"])

                st.session_state["gold_data"] = gold_df
                st.session_state["data_loaded"] = True

                return gold_df

            except Exception as e:
                st.sidebar.error(f"Error fetching data: {e}")
                return None

    if st.sidebar.button("Use Sample Data"):

        years = list(range(1964, datetime.datetime.now().year + 1))

        base_price = 100
        prices = [base_price]
        for i in range(1, len(years)):

            growth_factor = np.random.uniform(1.02, 1.15)

            if i % 7 == 0:
                growth_factor = np.random.uniform(0.9, 0.98)
            new_price = prices[-1] * growth_factor
            prices.append(new_price)

        sample_data = pd.DataFrame({"Year": years, "Price": prices})

        st.session_state["gold_data"] = sample_data
        st.session_state["data_loaded"] = True
        st.sidebar.success("✅ Using sample data")

        return sample_data

    if "data_loaded" in st.session_state and st.session_state["data_loaded"]:
        st.sidebar.success("✅ Data already loaded")
        return st.session_state["gold_data"]

    return None


def prepare_data(data_frame):
    if data_frame is not None:

        data_frame["Year"] = pd.to_datetime(data_frame["Year"], format="%Y")
        data_frame = data_frame.sort_values("Year")
        data_frame = data_frame.reset_index(drop=True)

        data_frame["YearValue"] = data_frame["Year"].dt.year

        return data_frame
    return None


def train_arima_model(time_series):

    st.sidebar.markdown("### Model Parameters")
    p = st.sidebar.slider("AR Parameter (p)", 0, 10, 5)
    d = st.sidebar.slider("Difference Order (d)", 0, 2, 1)
    q = st.sidebar.slider("MA Parameter (q)", 0, 10, 0)

    with st.spinner(f"Training ARIMA({p},{d},{q}) model..."):
        try:

            arima_model = ARIMA(time_series, order=(p, d, q))
            arima_results = arima_model.fit()
            st.sidebar.success("✅ Model trained successfully")
            return arima_results
        except Exception as e:
            st.sidebar.error(f"Error training model: {e}")
            return None


def predict_future_prices(arima_model, data_frame):
    if arima_model is not None and data_frame is not None:

        years_to_predict = st.sidebar.slider("Years to Predict", 1, 10, 5)

        with st.spinner(
            f"Generating predictions for the next {years_to_predict} years..."
        ):
            last_year = data_frame["YearValue"].max()
            future_years = np.array(
                range(last_year + 1, last_year + years_to_predict + 1)
            )

            try:
                arima_forecast = arima_model.forecast(steps=years_to_predict)
                st.sidebar.success("✅ Predictions generated")
                return future_years, arima_forecast
            except Exception as e:
                st.sidebar.error(f"Error generating predictions: {e}")
                return None, None
    return None, None


def create_visualizations(historical_df, future_years, arima_predictions):
    if (
        historical_df is not None
        and future_years is not None
        and arima_predictions is not None
    ):

        future_df = pd.DataFrame(
            {
                "Year": pd.to_datetime(future_years, format="%Y"),
                "ARIMA_Predicted": arima_predictions,
            }
        )

        tab1, tab2, tab3 = st.tabs(
            ["Full History & Predictions", "Recent Trends", "Statistics"]
        )

        with tab1:
            st.markdown(
                '<h2 class="subheader">Historical Gold Price & Future Predictions</h2>',
                unsafe_allow_html=True,
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=historical_df["Year"],
                    y=historical_df["Price"],
                    mode="lines",
                    name="Historical Gold Price",
                    line=dict(color="#B8860B", width=2),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=future_df["Year"],
                    y=future_df["ARIMA_Predicted"],
                    mode="lines+markers",
                    name="ARIMA Predictions",
                    line=dict(color="#FF4500", width=2),
                    marker=dict(size=8),
                )
            )

            for i, (year, arima_price) in enumerate(
                zip(future_years, future_df["ARIMA_Predicted"])
            ):
                fig.add_annotation(
                    x=pd.to_datetime(year, format="%Y"),
                    y=arima_price,
                    text=f"₹{arima_price:.2f}",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40,
                )

            fig.update_layout(
                title="Gold Price Historical Data and Future Predictions",
                xaxis_title="Year",
                yaxis_title="Gold Price (₹)",
                height=600,
                template="plotly_white",
                hovermode="x unified",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )

            fig.add_vline(
                x=historical_df["Year"].max(),
                line_width=1,
                line_dash="dash",
                line_color="gray",
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown(
                '<h2 class="subheader">Recent Gold Price Trends & Predictions</h2>',
                unsafe_allow_html=True,
            )

            recent_years = st.slider("Show recent years", 5, 20, 15)
            recent_df = historical_df.tail(recent_years)

            fig2 = go.Figure()

            fig2.add_trace(
                go.Scatter(
                    x=recent_df["Year"],
                    y=recent_df["Price"],
                    mode="lines",
                    name=f'Recent Gold Prices ({recent_df["YearValue"].min()}-{recent_df["YearValue"].max()})',
                    line=dict(color="#B8860B", width=2),
                )
            )

            fig2.add_trace(
                go.Scatter(
                    x=future_df["Year"],
                    y=future_df["ARIMA_Predicted"],
                    mode="lines+markers",
                    name="ARIMA Predictions",
                    line=dict(color="#FF4500", width=2),
                    marker=dict(size=8),
                )
            )

            for i, (year, arima_price) in enumerate(
                zip(future_years, future_df["ARIMA_Predicted"])
            ):
                fig2.add_annotation(
                    x=pd.to_datetime(year, format="%Y"),
                    y=arima_price,
                    text=f"₹{arima_price:.2f}",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40,
                )

            fig2.update_layout(
                title=f'Recent Gold Price Trends and Future Predictions ({recent_df["YearValue"].min()}-{future_years[-1]})',
                xaxis_title="Year",
                yaxis_title="Gold Price (₹)",
                height=600,
                template="plotly_white",
                hovermode="x unified",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )

            fig2.add_vline(
                x=historical_df["Year"].max(),
                line_width=1,
                line_dash="dash",
                line_color="gray",
            )

            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            st.markdown(
                '<h2 class="subheader">Statistics & Analysis</h2>',
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Predicted Gold Prices")

                display_df = pd.DataFrame(
                    {
                        "Year": future_years,
                        "ARIMA Prediction (₹)": [
                            f"₹{price:.2f}" for price in arima_predictions
                        ],
                    }
                )
                st.dataframe(display_df, use_container_width=True)

            with col2:
                st.subheader("Historical Statistics")

                historical_prices = historical_df["Price"].values
                mean_price = np.mean(historical_prices)
                median_price = np.median(historical_prices)
                min_price = np.min(historical_prices)
                max_price = np.max(historical_prices)

                years_elapsed = (
                    historical_df["YearValue"].max() - historical_df["YearValue"].min()
                )
                historical_cagr = (historical_prices[-1] / historical_prices[0]) ** (
                    1 / years_elapsed
                ) - 1

                arima_last_price = future_df["ARIMA_Predicted"].iloc[-1]
                prediction_years = len(future_df)
                arima_cagr = (arima_last_price / historical_prices[-1]) ** (
                    1 / prediction_years
                ) - 1

                stats = pd.DataFrame(
                    {
                        "Metric": [
                            "Mean Price",
                            "Median Price",
                            "Minimum Price",
                            "Maximum Price",
                            "Historical CAGR",
                            f"Predicted CAGR ({future_years[0]}-{future_years[-1]})",
                        ],
                        "Value": [
                            f"₹{mean_price:.2f}",
                            f"₹{median_price:.2f}",
                            f"₹{min_price:.2f} (Year: {historical_df.loc[historical_df['Price'] == min_price, 'YearValue'].values[0]})",
                            f"₹{max_price:.2f} (Year: {historical_df.loc[historical_df['Price'] == max_price, 'YearValue'].values[0]})",
                            f"{historical_cagr:.2%}",
                            f"{arima_cagr:.2%}",
                        ],
                    }
                )
                st.dataframe(stats, use_container_width=True, hide_index=True)

            st.subheader("Yearly Price Change (% Growth)")

            historical_df["Year_Change"] = historical_df["Price"].pct_change() * 100

            future_df_with_last_historical = pd.DataFrame(
                {
                    "Year": pd.concat(
                        [historical_df["Year"].tail(1), future_df["Year"]]
                    ),
                    "Price": pd.concat(
                        [historical_df["Price"].tail(1), future_df["ARIMA_Predicted"]]
                    ),
                }
            )
            future_df_with_last_historical["Year_Change"] = (
                future_df_with_last_historical["Price"].pct_change() * 100
            )

            fig3 = go.Figure()

            fig3.add_trace(
                go.Bar(
                    x=historical_df["Year"][1:],
                    y=historical_df["Year_Change"][1:],
                    name="Historical Yearly Change",
                    marker_color="#B8860B",
                )
            )

            fig3.add_trace(
                go.Bar(
                    x=future_df["Year"],
                    y=future_df_with_last_historical["Year_Change"][1:],
                    name="Predicted Yearly Change",
                    marker_color="#FF4500",
                )
            )

            fig3.update_layout(
                title="Yearly Gold Price Change (%)",
                xaxis_title="Year",
                yaxis_title="Price Change (%)",
                height=500,
                template="plotly_white",
                hovermode="x unified",
                barmode="group",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )

            fig3.add_hline(y=0, line_width=1, line_dash="solid", line_color="black")

            st.plotly_chart(fig3, use_container_width=True)

        return future_df
    return None


def calculate_roi(historical_df, future_df):
    if historical_df is not None and future_df is not None:
        st.markdown(
            '<h2 class="subheader">Investment ROI Calculator</h2>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:

            investment_amount = st.number_input(
                "Investment Amount (₹)",
                min_value=1000,
                max_value=10000000,
                value=100000,
                step=1000,
            )

            min_year = historical_df["YearValue"].min()
            max_year = historical_df["YearValue"].max()
            investment_year = st.slider(
                "Investment Year", min_year, max_year, max_year - 5
            )

            max_future_year = int(future_df["Year"].dt.year.max())
            sell_year = st.slider(
                "Sell Year", investment_year + 1, max_future_year, max_future_year
            )

        try:

            investment_price = historical_df.loc[
                historical_df["YearValue"] == investment_year, "Price"
            ].values[0]

            if sell_year <= max_year:
                sell_price = historical_df.loc[
                    historical_df["YearValue"] == sell_year, "Price"
                ].values[0]
                price_source = "Historical"
            else:
                sell_price = future_df.loc[
                    future_df["Year"].dt.year == sell_year, "ARIMA_Predicted"
                ].values[0]
                price_source = "Predicted"

            gold_quantity = investment_amount / investment_price

            sell_value = gold_quantity * sell_price

            absolute_return = sell_value - investment_amount
            percentage_return = (sell_value / investment_amount - 1) * 100

            years_held = sell_year - investment_year
            annualized_return = (
                (sell_value / investment_amount) ** (1 / years_held) - 1
            ) * 100

            with col2:
                st.subheader("Investment Results")

                metrics_col1, metrics_col2 = st.columns(2)

                with metrics_col1:
                    st.metric(
                        label="Investment Amount", value=f"₹{investment_amount:,.2f}"
                    )
                    st.metric(
                        label=f"Gold Price ({investment_year})",
                        value=f"₹{investment_price:,.2f}",
                    )
                    st.metric(
                        label="Gold Quantity", value=f"{gold_quantity:,.4f} units"
                    )

                with metrics_col2:
                    st.metric(
                        label=f"Value in {sell_year} ({price_source})",
                        value=f"₹{sell_value:,.2f}",
                    )
                    st.metric(
                        label="Total Return",
                        value=f"₹{absolute_return:,.2f}",
                        delta=f"{percentage_return:,.2f}%",
                    )
                    st.metric(
                        label="Annualized Return", value=f"{annualized_return:,.2f}%"
                    )

                if percentage_return > 0:
                    st.success(
                        f"💰 Your investment would grow by {percentage_return:.2f}% over {years_held} years!"
                    )
                else:
                    st.error(
                        f"📉 Your investment would decrease by {abs(percentage_return):.2f}% over {years_held} years!"
                    )
        except Exception as e:
            st.error(f"Error calculating ROI: {e}")


def main():
    gold_data = scrape_historical_data()

    if gold_data is not None:
        prepared_data = prepare_data(gold_data)
        if st.sidebar.checkbox("Show Raw Data"):
            st.subheader("Raw Gold Price Data")
            st.dataframe(gold_data)

        if prepared_data is not None:
            arima_model = train_arima_model(prepared_data["Price"])
            if arima_model is not None:
                future_years, arima_predictions = predict_future_prices(
                    arima_model, prepared_data
                )
                if future_years is not None and arima_predictions is not None:
                    future_df = create_visualizations(
                        prepared_data, future_years, arima_predictions
                    )
                    if future_df is not None:
                        calculate_roi(prepared_data, future_df)
    else:
        st.info(
            "👈 Please use the sidebar to fetch gold price data or use sample data."
        )


def add_footer():
    st.markdown(
        """
    <hr style="margin-top: 50px;"/>
    <div style="text-align: center; color: gray; font-size: 0.85rem; line-height: 1.6;">
        <p><strong>Gold Price Analysis & Prediction App</strong></p>
        <p>Built with <strong>Python</strong>, <strong>Streamlit</strong>, and <strong>ARIMA modeling</strong></p>
        <p>Gold price data sourced from <a href="https://groww.in" target="_blank" style="color: inherit; text-decoration: underline;">Groww.in</a></p>
        <p>Created by Amit Sharma</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
    add_footer()
