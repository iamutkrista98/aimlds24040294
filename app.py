import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import xgboost as xgb
import pickle
import os
import warnings

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Kalimati Price Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for dark mode
st.markdown("""
<style>
    .metric-card {
        background: #2a2a2a;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
        color: #e0e0e0;
    }
    .metric-card:hover {
        transform: translateY(-3px);
    }
    .sidebar .sidebar-content {
        background-color: #1e1e1e;
        color: #e0e0e0;
    }
    .stSelectbox > div > div > div {
        background-color: #333333;
        border-radius: 5px;
        border: 1px solid #555555;
        color: #e0e0e0;
    }
    .stSlider > div > div > div {
        background-color: #333333;
    }
    .stRadio > label {
        color: #e0e0e0;
    }
    .stMultiselect > div > div > div {
        background-color: #333333;
        color: #e0e0e0;
    }
    body {
        background-color: #121212;
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


# Data loading and preprocessing
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("top_30_commodities.csv")
    except FileNotFoundError:
        st.error("Dataset file not found. Please check the path.")
        return pd.DataFrame()

    df = df.copy()
    df['Commodity'] = df['Commodity'].str.strip().str.title()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    numeric_cols = ['Minimum', 'Maximum', 'Average']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=numeric_cols)

    df = df[(df['Minimum'] <= df['Maximum']) &
            (df['Average'].between(df['Minimum'], df['Maximum']))]

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month_name()
    df['MonthNum'] = df['Date'].dt.month
    df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)

    return df


# Visualization data preparation
def prepare_visualization_data(df, product, aggregation='Monthly'):
    product_df = df[df['Commodity'] == product].sort_values('Date')

    if aggregation == "Daily":
        data = product_df.groupby('Date').agg({
            'Average': 'mean',
            'Maximum': 'max',
            'Minimum': 'min'
        }).reset_index()
        data['6M_MA'] = data['Average'].rolling(window=180, min_periods=1).mean()
        data['12M_MA'] = data['Average'].rolling(window=360, min_periods=1).mean()

    elif aggregation == "Monthly":
        data = product_df.groupby(pd.Grouper(key='Date', freq='M')).agg({
            'Average': 'mean',
            'Maximum': 'max',
            'Minimum': 'min'
        }).reset_index()
        data['6M_MA'] = data['Average'].rolling(window=6, min_periods=1).mean()
        data['12M_MA'] = data['Average'].rolling(window=12, min_periods=1).mean()

    else:  # Quarterly
        data = product_df.groupby(pd.Grouper(key='Date', freq='Q')).agg({
            'Average': 'mean',
            'Maximum': 'max',
            'Minimum': 'min'
        }).reset_index()
        data['6M_MA'] = data['Average'].rolling(window=2, min_periods=1).mean()
        data['12M_MA'] = data['Average'].rolling(window=4, min_periods=1).mean()

    return data


# Safe name function for model files
def create_safe_name(commodity):
    safe = commodity.replace('/', '_')
    return safe


# Prediction generation
@st.cache_data
def generate_predictions(commodity, n_days, model_choice):
    df_commodity = df[df['Commodity'] == commodity].copy()
    df_commodity.sort_values('Date', inplace=True)
    ts_data = df_commodity[['Date', 'Average', 'Minimum', 'Maximum']].dropna()
    ts_data.set_index('Date', inplace=True)

    try:
        scaled_data = scaler.transform(ts_data[['Average']])
    except Exception as e:
        st.error(f"Error scaling data: {e}")
        return None, None, None, None

    lstm_future_predictions = None
    xgb_future_predictions = None
    ensemble_future_predictions = None
    look_back = 365

    last_date = ts_data.index.max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=n_days, freq='D')

    safe_name = create_safe_name(commodity)
    lstm_model_path = f'lstm_model_{safe_name}.keras'
    xgb_model_path = f'xgb_model_{safe_name}.json'

    if model_choice in ["LSTM", "Ensemble"]:
        if not os.path.exists(lstm_model_path):
            st.error(f"LSTM model file '{lstm_model_path}' not found.")
            return None, None, None, None
        try:
            lstm_model = load_model(lstm_model_path)
            last_sequence = scaled_data[-look_back:]
            current_sequence = last_sequence.copy()
            lstm_future_pred = []
            for _ in range(n_days):
                reshaped_seq = current_sequence.reshape((1, look_back, 1))
                next_scaled = lstm_model.predict(reshaped_seq, verbose=0)
                lstm_future_pred.append(next_scaled[0, 0])
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = next_scaled
            lstm_future_predictions = scaler.inverse_transform(np.array(lstm_future_pred).reshape(-1, 1)).flatten()
        except Exception as e:
            st.error(f"Error generating LSTM predictions: {e}")
            return None, None, None, None

    if model_choice in ["XGBoost", "Ensemble"]:
        if not os.path.exists(xgb_model_path):
            st.error(f"XGBoost model file '{xgb_model_path}' not found.")
            return None, None, None, None
        try:
            xgb_model = xgb.XGBRegressor()
            xgb_model.load_model(xgb_model_path)
            agg_min = ts_data['Minimum'].mean()
            agg_max = ts_data['Maximum'].mean()
            xgb_features = []
            for future_date in future_dates:
                xgb_features.append({
                    'Year': future_date.year,
                    'Month': future_date.month,
                    'Day': future_date.day,
                    'DayOfWeek': future_date.dayofweek,
                    'Quarter': (future_date.month - 1) // 3 + 1,
                    'IsWeekend': 1 if future_date.dayofweek >= 5 else 0,
                    'Minimum': agg_min,
                    'Maximum': agg_max
                })
            xgb_df = pd.DataFrame(xgb_features)
            xgb_future_predictions = xgb_model.predict(xgb_df)
        except Exception as e:
            st.error(f"Error generating XGBoost predictions: {e}")
            return None, None, None, None

    if model_choice == "Ensemble" and lstm_future_predictions is not None and xgb_future_predictions is not None:
        ensemble_future_predictions = (lstm_future_predictions + xgb_future_predictions) / 2

    return future_dates, lstm_future_predictions, xgb_future_predictions, ensemble_future_predictions


# Load data
df = load_data()
if df.empty:
    st.error("Failed to load dataset. Please verify data source.")
    st.stop()

# Sidebar navigation and filters
with st.sidebar:
    st.title("Kalimati Market Analytics")
    page = st.radio("Navigate", ["Price Trends", "Price Predictions"])

    st.markdown("### Filter Options")
    selected_commodity = st.selectbox(
        "Select Commodity",
        options=sorted(df['Commodity'].unique()),
        index=sorted(df['Commodity'].unique()).index("Tomato Big(Nepali)") if "Tomato Big(Nepali)" in df[
            'Commodity'].unique() else 0,
        help="Select a commodity to analyze"
    )

    if page == "Price Trends":
        time_range = st.slider(
            "Select Time Range (Years)",
            min_value=int(df['Year'].min()),
            max_value=int(df['Year'].max()),
            value=(int(df['Year'].min()), int(df['Year'].max())),
            step=1
        )
        aggregation = st.selectbox(
            "Time Aggregation",
            options=["Daily", "Monthly", "Quarterly"],
            index=1
        )
    else:
        period_options = {
            "1 Week (7 days)": 7,
            "1 Month (30 days)": 30,
            "1 Year (365 days)": 365,
            "2 Years (730 days)": 730
        }
        period_label = st.selectbox("Select Prediction Period", list(period_options.keys()))
        n_days = period_options[period_label]

        safe_name = create_safe_name(selected_commodity)
        model_options = []
        lstm_model_path = f'lstm_model_{safe_name}.keras'
        xgb_model_path = f'xgb_model_{safe_name}.json'
        if os.path.exists(lstm_model_path):
            model_options.append("LSTM")
        if os.path.exists(xgb_model_path):
            model_options.append("XGBoost")
        if all(os.path.exists(f) for f in [lstm_model_path, xgb_model_path]):
            model_options.append("Ensemble")
        if not model_options:
            st.error(
                f"No models found for {selected_commodity}. Ensure model files '{lstm_model_path}' or '{xgb_model_path}' exist.")
            st.stop()
        model_choice = st.selectbox("Select Model", model_options)

# Load scaler
try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'scaler.pkl' not found in the directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading scaler: {e}")
    st.stop()

# Main interface
if selected_commodity:
    if page == "Price Trends":
        st.title("🌾 Kalimati Market Price Analysis")
        st.markdown(f"### Price Trends for {selected_commodity}")

        df_filtered = df[(df['Year'] >= time_range[0]) & (df['Year'] <= time_range[1])]
        viz_data = prepare_visualization_data(df_filtered, selected_commodity, aggregation)

        st.markdown("### Key Statistics")
        cols = st.columns(3)

        with cols[0]:
            peak_price = viz_data['Maximum'].max()
            peak_date = viz_data.loc[viz_data['Maximum'].idxmax(), 'Date']
            st.markdown(f"""
            <div class="metric-card">
                <h3>📈 Peak Price</h3>
                <p style="font-size: 24px; margin: 0.5rem 0;">NPR {peak_price:.2f}</p>
                <small>{peak_date.strftime('%b %Y')}</small>
            </div>
            """, unsafe_allow_html=True)

        with cols[1]:
            low_price = viz_data['Minimum'].min()
            low_date = viz_data.loc[viz_data['Minimum'].idxmin(), 'Date']
            st.markdown(f"""
            <div class="metric-card">
                <h3>📉 Lowest Price</h3>
                <p style="font-size: 24px; margin: 0.5rem 0;">NPR {low_price:.2f}</p>
                <small>{low_date.strftime('%b %Y')}</small>
            </div>
            """, unsafe_allow_html=True)

        with cols[2]:
            current_avg = viz_data.iloc[-1]['Average']
            current_date = viz_data.iloc[-1]['Date']
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔄 Current Average</h3>
                <p style="font-size: 24px; margin: 0.5rem 0;">NPR {current_avg:.2f}</p>
                <small>{current_date.strftime('%b %Y')}</small>
            </div>
            """, unsafe_allow_html=True)

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.15,
            subplot_titles=(
                "Price Variations and Range",
                "Long-term Trend Analysis"
            )
        )

        colors = {
            'avg': '#1abc9c',  # Vibrant teal
            'max': 'rgba(39, 174, 96, 0.7)',  # Bright green with a soft opacity
            'min': 'rgba(52, 152, 219, 0.5)',  # Cool blue with moderate transparency
            'ma6': '#9b59b6',  # Rich purple
            'ma12': '#34495e',  # Subtle, elegant navy
        }

        fig.add_trace(go.Scatter(
            x=viz_data['Date'],
            y=viz_data['Maximum'],
            name='Maximum',
            line=dict(color=colors['max'], dash='dash'),
            mode='lines',
            showlegend=True
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=viz_data['Date'],
            y=viz_data['Minimum'],
            name='Minimum',
            line=dict(color=colors['min']),
            fill='tonexty',
            fillcolor=colors['min'],
            mode='lines',
            showlegend=True
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=viz_data['Date'],
            y=viz_data['Average'],
            name='Average',
            line=dict(color=colors['avg'], width=2.5),
            mode='lines+markers',
            marker=dict(size=6, symbol='circle'),
            hovertemplate="<b>%{x|%b %Y}</b><br>" +
                          f"Average: NPR %{{y:.2f}}<br>" +
                          "Range: NPR %{text}",
            text=[f"{row['Minimum']:.2f}-{row['Maximum']:.2f}" for _, row in viz_data.iterrows()],
            showlegend=True
        ), row=1, col=1)

        if '6M_MA' in viz_data.columns:
            fig.add_trace(go.Scatter(
                x=viz_data['Date'],
                y=viz_data['6M_MA'],
                name='6M Trend',
                line=dict(color=colors['ma6'], width=2),
                showlegend=True
            ), row=2, col=1)

        if '12M_MA' in viz_data.columns:
            fig.add_trace(go.Scatter(
                x=viz_data['Date'],
                y=viz_data['12M_MA'],
                name='12M Trend',
                line=dict(color=colors['ma12'], width=3),
                showlegend=True
            ), row=2, col=1)

        fig.update_layout(
            height=900,
            template='plotly_dark',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.05,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(0,0,0,0.7)',
                bordercolor='rgba(255,255,255,0.1)',
                borderwidth=1,
                font=dict(size=12, color='#e0e0e0')
            ),
            margin=dict(t=100, b=50, l=50, r=50),
            xaxis=dict(
                tickformat="%b %Y",
                rangeslider=dict(visible=True),
                type="date",
                gridcolor='rgba(255,255,255,0.1)',
                tickcolor='#e0e0e0',
                tickfont=dict(color='#e0e0e0')
            ),
            yaxis=dict(
                title=dict(text="Price (NPR)", font=dict(color='#e0e0e0')),
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='#e0e0e0')
            ),
            yaxis2=dict(
                title=dict(text="Price (NPR)", font=dict(color='#e0e0e0')),
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='#e0e0e0')
            ),
            showlegend=True,
            plot_bgcolor='rgba(30,30,30,1)',
            paper_bgcolor='rgba(18,18,18,1)'
        )

        fig.update_xaxes(
            tickangle=45,
            tickformat="%b %Y",
            row=1, col=1,
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#e0e0e0')
        )

        fig.update_xaxes(
            tickangle=45,
            tickformat="%b %Y",
            row=2, col=1,
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#e0e0e0')
        )

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("View Detailed Data"):
            st.markdown(f"#### {selected_commodity}")
            st.dataframe(
                viz_data.sort_values('Date', ascending=False),
                column_config={
                    'Date': st.column_config.DateColumn(format="YYYY-MM"),
                    'Average': st.column_config.NumberColumn(format="NPR %.2f"),
                    'Maximum': st.column_config.NumberColumn(format="NPR %.2f"),
                    'Minimum': st.column_config.NumberColumn(format="NPR %.2f"),
                    '6M_MA': st.column_config.NumberColumn("6-Month Avg", format="NPR %.2f"),
                    '12M_MA': st.column_config.NumberColumn("12-Month Avg", format="NPR %.2f")
                },
                hide_index=True,
                height=400
            )

    elif page == "Price Predictions":
        st.title("📅 Price Predictions")
        st.markdown(f"### Future Price Predictions for {selected_commodity}")

        future_dates, lstm_pred, xgb_pred, ensemble_pred = generate_predictions(selected_commodity, n_days,
                                                                                model_choice)
        if future_dates is None:
            st.stop()

        predictions_df = pd.DataFrame({'Date': future_dates})
        if model_choice == "LSTM":
            predictions_df['Predicted_Average_Price'] = lstm_pred
        elif model_choice == "XGBoost":
            predictions_df['Predicted_Average_Price'] = xgb_pred
        elif model_choice == "Ensemble":
            predictions_df['Predicted_Average_Price'] = ensemble_pred

        st.markdown(f"#### Predicted Prices for the Next {period_label} (Model: {model_choice})")
        st.dataframe(
            predictions_df,
            column_config={
                'Date': st.column_config.DateColumn(format="YYYY-MM-DD"),
                'Predicted_Average_Price': st.column_config.NumberColumn(format="NPR %.2f")
            },
            hide_index=True
        )

        historical_data = df[df['Commodity'] == selected_commodity].sort_values('Date')
        historical_data = historical_data.groupby('Date').agg({'Average': 'mean'}).reset_index()
        historical_data = historical_data.tail(30)

        fig_pred = go.Figure()

        fig_pred.add_trace(go.Scatter(
            x=historical_data['Date'],
            y=historical_data['Average'],
            name='Historical Average',
            line=dict(color='#1abc9c', width=2),
            mode='lines+markers',
            marker=dict(size=6, symbol='circle')
        ))

        fig_pred.add_trace(go.Scatter(
            x=predictions_df['Date'],
            y=predictions_df['Predicted_Average_Price'],
            name='Predicted Average',
            line=dict(color='#e74c3c', width=2),
            mode='lines+markers',
            marker=dict(size=8, symbol='diamond')
        ))

        fig_pred.update_layout(
            title=f'Price Trend and Predictions for {selected_commodity}',
            height=600,
            template='plotly_dark',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.05,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(0,0,0,0.7)',
                bordercolor='rgba(255,255,255,0.1)',
                borderwidth=1,
                font=dict(size=12, color='#e0e0e0')
            ),
            xaxis=dict(
                title="Date",
                tickformat="%b %d, %Y",
                type="date",
                gridcolor='rgba(255,255,255,0.1)',
        tickcolor = '#e0e0e0',
        tickfont = dict(color='#e0e0e0')
        ),
        yaxis = dict(
            title="Price (NPR)",
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#e0e0e0')
        ),
        plot_bgcolor = 'rgba(30,30,30,1)',
        paper_bgcolor = 'rgba(18,18,18,1)'
        )

        st.plotly_chart(fig_pred, use_container_width=True)

        st.download_button(
            label="Download Predictions as CSV",
            data=predictions_df.to_csv(index=False),
            file_name=f"predictions_{safe_name}_{model_choice.lower()}_{n_days}days.csv",
            mime="text/csv"
        )

        st.markdown("""
        ### About the Models
        - **LSTM**: A recurrent neural network model suited for time-series forecasting, capturing sequential patterns.
        - **XGBoost**: A gradient boosting model that uses engineered features like date components for regression.
        - **Ensemble**: Combines LSTM and XGBoost predictions by averaging them for improved robustness.
        """)
else:
    st.error("No commodity selected. Please select a commodity from the sidebar.")