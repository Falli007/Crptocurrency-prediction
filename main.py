import yfinance as yf
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import randint
from PIL import Image

import streamlit as st
import sys
import subprocess

import os
import joblib
from datetime import datetime
import time
import mplfinance as mpf
import warnings
warnings.filterwarnings('ignore')

# Set page width
st.markdown(
    """
    <style>
    .reportview-container {
        max-width: 1000px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# To add a colorful header
st.markdown(
    """
    <h1 style='text-align: center; color: #4287f5;'>SOLiGence Cryptocurrency App</h1>
    """,
    unsafe_allow_html=True
)

# Define the directory to save the uploaded image
image_directory = "C:/Users/allif/Downloads"
image_filename = "bitcoin-price-prediction.png"
image_path = os.path.join(image_directory, image_filename)

# Display file uploader if an image has not been uploaded yet
if "uploaded_image" not in st.session_state:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Check if a file was uploaded
    if uploaded_file is not None:
        # Process the uploaded file
        image = Image.open(uploaded_file)

        # Save the image to the specified directory
        image.save(image_path)

        # Save the image path to session state
        st.session_state.uploaded_image = image_path

# Display the uploaded image
if "uploaded_image" in st.session_state:
    image_path = st.session_state.uploaded_image
    image = Image.open(image_path)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save the image
    if st.button("Save Image"):
        saved_image_path = os.path.join(image_directory, "saved_image.png")
        image.save(saved_image_path)
        st.success("Image saved successfully.")

# To add a colorful progress bar
progress = st.progress(0)
for i in range(100):
    time.sleep(0.01)
    progress.progress(i + 1)

# To add a custom footer
st.markdown(
    """
    <footer style='text-align: center; padding-top: 2rem;'>
        Made with ❤️ by Fredrick Alli
    </footer>
    """,
    unsafe_allow_html=True
)

coins = "BTC,ETH,ADA,BNB,XRP,LTC,BCH,DOGE,XLM,AAVE,DASH,NEXO,ETC,XMR,WAVES,MIOTA,ICX,DOT,EOS,SOL"
coin_list = coins.split(",")  # ['BTC', ' ETH'...]
coin_dictionary = {coin: coin_list.index(coin) for coin in coin_list}

# load model, encoder, and scaler
scaler_ = joblib.load('myscaler.joblib')
label_encoder_ = joblib.load('mylabel_encoder.joblib')
model_ = joblib.load('mymodel.joblib')

# define functions
def myScaler_prod(data):
    global scaler_
    data_ = data.copy()
    data_scaled_ = scaler_.transform(data_)
    data_scaled_ = pd.DataFrame(data_scaled_, columns=data_.columns)
    data_scaled_.index = data_.index
    return data_scaled_


def myInverseScale_prod(data):
    global scaler_
    data_scaler_inverse = scaler_.inverse_transform(data)
    data_scaler_inverse = pd.DataFrame(data_scaler_inverse, columns=data.columns)
    data_scaler_inverse.index = data.index
    return data_scaler_inverse

def getcoindata(coin_name_, interval_="1d"):
    crypto2 = yf.Ticker(f"{coin_name_}-USD").history(start='2018-01-01', end=str(datetime.today().date()),
                                                     interval=interval_)
    crypto2 = pd.DataFrame(crypto2)
    crypto2['crypto_name'] = coin_name_
    return crypto2

# Function to calculate the moving average
def calculate_moving_average(data, window):
    return data['Close'].rolling(window=window).mean()

# Function to calculate the profit price
def calculate_profit_price(starting_price, target_price):
    return round(starting_price + (target_price - starting_price) * 1.01, 2)

# Function to calculate the confidence level
def calculate_confidence_level(predicted_value, target_price):
    return round((predicted_value / target_price) * 100, 2)

def calculate_best_trade(data):
    max_profit = 0
    best_time_to_purchase = None
    best_time_to_sell = None

    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            purchase_price = data.iloc[i]['Close']
            sell_price = data.iloc[j]['Close']
            profit = sell_price - purchase_price

            if profit > max_profit:
                max_profit = profit
                best_time_to_purchase = data.index[i]
                best_time_to_sell = data.index[j]

    return best_time_to_purchase, best_time_to_sell, max_profit

def calculate_future_trade(data):
    last_price = data.iloc[-1]['Close']
    future_price = last_price * 1.01  # Assume a 1% increase in price
    purchase_price = last_price
    sell_price = calculate_profit_price(purchase_price, future_price)
    profit = sell_price - purchase_price

    return sell_price, profit

def execute_program(coin="ETH", time_string="1d"):
    global model_
    # fetch data
    df_test = getcoindata(coin_name_=coin, interval_=time_string)

    # label encoding
    df_test["crypto_encoded"] = label_encoder_.transform(df_test["crypto_name"])

    # scale
    to_scale_ = df_test[["Open", "High", "Low", "Close", "crypto_encoded"]]
    df_scaled_ = myScaler_prod(to_scale_)

    # predict
    predicted_ = model_.predict(df_scaled_[["Open", "High", "Low", "crypto_encoded"]])

    # inverse scale
    inverse_predicted = df_scaled_.copy()
    inverse_predicted["Close"] = predicted_
    inverse_predicted = myInverseScale_prod(inverse_predicted[["Open", "High", "Low", "Close", "crypto_encoded"]])

    # merge
    df_test["predicted"] = inverse_predicted["Close"]

    # Calculate moving average
    df_test["Moving Average"] = df_test["Close"].rolling(window=10).mean()

    # return dataset
    return df_test

def display_top_stories():
    stories = [
        {
            "title": "Top 4 new cryptocurrencies 2023",
            "description": "New Cryptocurrencies 2023",
            "source": "CoinsPaid Media",
            "published_at": "Updates every hour",
            "url": "https://coinspaidmedia.com/business/top-4-new-cryptocurrencies-2023/"
        },
        {
            "title": "Cryptocurrency News",
            "description": "Latest Crypto and Bitcoin News",
            "source": "Yahoo finance",
            "published_at": "Updates every hour",
            "url": "https://finance.yahoo.com/topic/crypto/?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8"
                   "&guce_referrer_sig=AQAAAM2TswRjGJmFYQ0QHIb7oLhP_5RDjIAkGGf76b2bj-tn1nCR6XLhO2RB3Olx25ftGcoOiMTJtLZ"
                   "eAU5QEoAiaWhmJEmDIn5Fa8iNX9cCkd-98M-CWIsZw8AWRIBBv72f4YQQwaGEmiVlYjevU7-ueW2hvJIrHCFAayhkJI3lterf"
        },
        {
            "title": "Cryptocurrency News",
            "description": "Insights into the biggest event shaping crypto industry",
            "source": "Coin Market",
            "published_at": "Updates every hour",
            "url": "https://coinmarketcap.com/headlines/news/"
        },
]
    for story in stories:
        st.write("Title:", story['title'])
        st.write("Description:", story['description'])
        st.write("Source:", story['source'])
        st.write("Published:", story['published_at'])
        st.write("URL:", story['url'])
        st.write("---------------")

# streamlit sidebar
with st.sidebar:
    st.header("OPTIONS")
    coin_list = ["BTC", "ETH", "XRP", "LTC", "BCH", "DOGE", "XLM", "AAVE", "DASH", "NEXO", "ETC",
                 "XMR", "WAVES", "MIOTA", "ICX", "DOT", "EOS", "SOL"]
    select_options = st.multiselect('Select Coins', coin_list)

    interval_options = {
        '1d': '1 Day',
        '1wk': '1 Week',
        '1mo': '1 Month',
        '3mo': '3 Months',
    }
    interval = st.selectbox('Select timeframe', tuple(interval_options.keys()), format_func=lambda x: interval_options[x])

    # Number of coins to buy or sell
    num_coins = st.number_input("Number of coins to buy/sell:", min_value=0.0, step=0.01, value=0.0)

    submit_button = st.button('Submit', key='submit_button', use_container_width=True)

# Load the trained model, encoder, and scaler
scaler_ = joblib.load('myscaler.joblib')
label_encoder_ = joblib.load('mylabel_encoder.joblib')
model_ = joblib.load('mymodel.joblib')

if submit_button:
    with st.spinner('Loading ☕...........'):

        data = execute_program(coin=select_options[0], time_string=interval)
        # To replace last close Data with the predicted
        data.at[data.index[-1], 'Close'] = data.iloc[-1]["predicted"]
        starting_price = data.iloc[-1]["Open"]
        target_price = data.iloc[-1]["Close"]
        profit_price = calculate_profit_price(starting_price, target_price)
        predicted_value = target_price  # To set the predicted value to the target price
        confidence_level = calculate_confidence_level(predicted_value, target_price)
        time.sleep(5)
        st.success('Done ✅')
        st.write("The last candlestick(yellow) is the predicted data.")
        st.write(f"Starting price => {round(starting_price, 2)} USD, Target price => {round(target_price, 2)} USD")
        st.write(f"Predicted value => {round(predicted_value, 2)} USD")
        st.write(f"Confidence Level: {confidence_level}")

        # Code to print the profit price
        st.write(f"Profit price => {profit_price} USD")

        # Find the best time to purchase and sell
        best_time_to_purchase, best_time_to_sell, max_profit = calculate_best_trade(data)

        # Display the best time to purchase and sell
        st.write(f"Best time to purchase: {best_time_to_purchase}")
        st.write(f"Best time to sell: {best_time_to_sell}")
        st.write(f"Anticipated profit: {max_profit} USD")

    # To plot
    # Customize market colors, and place inside style you intend to use:
    mc = mpf.make_marketcolors(up='yellow', down='yellow')
    s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)

    # Create an all `nan` values dataframe, same size, and same index as the first:
    nans = [float('nan')] * len(data)
    cdf = pd.DataFrame(dict(Open=nans, High=nans, Low=nans, Close=nans), index=data.index)

    # Copy in the specific candles that you want to color:
    cdf.loc[str(data.tail(1).index[0])] = data.loc[str(data.tail(1).index[0])]

    # Call `mpf.plot()` twice with the two dataframes and two styles:
    fig, ax1 = mpf.plot(data[-30:], type='candle', style='yahoo', returnfig=True)
    mpf.plot(cdf[-30:], type='candle', style=s, ax=ax1[0])
    st.pyplot(fig)

    # Plotting actual, predicted, and moving average
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data['Close'], color='blue', label='Actual Close Price')
    ax.plot(data.index, data['predicted'], color='red', label='Predicted Close Price')
    ax.plot(data.index, data['Moving Average'], color='green', label='Moving Average')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.set_title(f'{select_options[0]} Actual, Predicted, and Moving Average Close Price')
    ax.legend()

    # Display line chart
    st.pyplot(fig)

    # Display table
    st.write(data[-30:])

#To display top stories
    display_top_stories()
