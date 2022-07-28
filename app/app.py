import streamlit as st
import pandas as pd
import numpy as np
from itertools import cycle
import os

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# For reading stock data from yahoo
from pandas_datareader.data import DataReader
import yfinance as yf

# For time stamps
from datetime import datetime

# Sidebar
st.sidebar.title("Stock Price Prediction App")
directory = os.listdir(".")
filename = st.sidebar.selectbox("Select a dataset", directory)
scaler = MinMaxScaler(feature_range=(0, 1))

st.header("Stock Price Prediction")


def get_unique_date(date):

    list_of_unique_years = []

    unique_years = set(date)

    for date in unique_years:
        list_of_unique_years.append(date)

    return list_of_unique_years


# Get Data
# TODO: Load data from input and set it up for prediction
# Reading the dataset
@st.cache
def get_data(filename):
    if filename != "google.csv":
        return "./" + filename
    else:
        return "./google.csv"


df = pd.read_csv(filename)

st.markdown("**Below is a quick look at the dataset**")
st.write(df)

# filter out only the Close column
df = df.filter(["Date", "Close"])
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index(["Date"])

# Convert the dataframe to a numpy array
dataset = df.Close.values

# Get the number of rows to train the model on
data_len = int(np.ceil(len(dataset.reshape(-1, 1)) * 0.95))


# # scale the data
scaled_data = scaler.fit_transform(dataset.reshape(-1, 1))

# Create the testing data set
test_data = scaled_data[data_len - 60 :, :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[data_len:]
for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60 : i, 0])

# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Load model and predict on the data
from keras.models import load_model

model = load_model("./model")

# Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Plot the data
train = df[:data_len]
valid = df[data_len:]
valid["Predictions"] = predictions

original_title = 'The graph below shows the existing price trend (<b style="color:Blue; font-size: 15px;">blue</b>) of the provided dataset and the predicted trend (<b style="color:red; font-size: 15px;">red</b>).'
st.write(original_title, unsafe_allow_html=True)

# Visualize the data
plt.figure(figsize=(16, 6))
plt.title("LSTM Price Prediction")
plt.xlabel("Date", fontsize=18)
plt.ylabel("Close Price USD ($)", fontsize=18)
plt.plot(train["Close"])
plt.plot(valid[["Predictions"]])
plt.legend(["Train", "Predictions"], loc="lower left")

st.pyplot(fig=plt)
