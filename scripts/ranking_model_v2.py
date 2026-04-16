from pyspark.sql import functions as F, SparkSession
from pyspark.ml.functions import vector_to_array
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns


def forecast_revenue(df):
    """
    Forecasts merchant revenue 3 periods ahead using a stacked LSTM neural network.
    Trains on lag features derived from historical revenue and growth rates.
    Returns a list of 3 forecasted revenue values.
    """
    features = ["revenue_lag_1", "revenue_lag_2", "revenue_lag_3", "revenue_growth_lag_1", "revenue_growth_lag_2"]

    # 80/20 train-test split; seed fixed for reproducibility
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=30032)

    # Reshape to (samples, timesteps, features=1) as required by Keras LSTM
    X_train = np.array(train_df.select(features).collect())
    y_train = np.array(train_df.select("revenue").collect())
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    # Stacked LSTM: three layers capture short, medium, and longer-range patterns
    forecaster = Sequential()
    forecaster.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))  # Layer 1
    forecaster.add(LSTM(units=50, return_sequences=True, activation='relu'))                                     # Layer 2
    forecaster.add(LSTM(units=32, activation='relu'))                                                            # Layer 3 (output)
    forecaster.add(Dense(1))

    forecaster.compile(optimizer='rmsprop', loss='mse')
    forecaster.fit(X_train, y_train, epochs=200, verbose=0, shuffle=False)

    last_values = X_train[-1].reshape(1, X_train.shape[1], 1)
    input_values = generate_input(inputCol=last_values, prediction=None, period=0, df=df)

    predictions = []
    print(f"Initial input values: {input_values}")
    for i in range(1, 4):  # Predict 3 periods ahead
        prediction = forecaster.predict(input_values)
        predictions.append(prediction[0][0])
        input_values = generate_input(inputCol=input_values, prediction=prediction, period=i, df=df)

    return predictions


def generate_input(inputCol, prediction, period, df):
    """
    Prepares the rolling input window for the next LSTM forecast step.
    Shifts existing lag values and injects either the latest actual revenue (period 0)
    or the most recent prediction as the new lag-1 value.
    Returns the updated input array ready for the next forecast call.
    """
    new_input = inputCol.copy()

    # Shift lag values backward: lag_2 → lag_3, lag_1 → lag_2
    new_input[0][2] = new_input[0][1]
    new_input[0][1] = new_input[0][0]

    if period == 0:
        # Seed the first forecast with the last known actual revenue
        new_input[0][0] = df.select('revenue').tail(1)[0][0]
    else:
        # Use the latest prediction as the new lag-1 for the next step
        new_input[0][0] = prediction[0][0]

    # Shift growth lag and recompute the most recent growth rate
    new_input[0][4] = new_input[0][3]
    new_input[0][3] = (new_input[0][0] - new_input[0][1]) / new_input[0][1]

    return new_input


def generate_num_order_weight(mean, observed):
    """
    Computes a sigmoid-based weight for merchant order volume.
    Returns a value in (0, 1) — higher weight for merchants with order volumes above the mean.
    """
    return (1 / (1 + np.exp(-(observed - mean))))
