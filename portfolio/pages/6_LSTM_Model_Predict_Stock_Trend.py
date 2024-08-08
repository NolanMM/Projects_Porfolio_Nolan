import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
import streamlit as st
from Project.Project_6.Training_Model import StockPricePredictor

css_file = "./styles/main.css"
IMAGE_TRAINING_PROCESSING = "./assets/project_6/Training_Processing.png"
IMAGE_MODEL_SUMMARY ="./assets/project_6/Model_Summary.png"
IMAGE_RESULT ="./assets/project_6/Results.png"
MODEL_SAVE_PATH = "./Project/Project_6/keras_model.keras"
SAMPLE_FILE = "./Project/Project_6/Training_Model_Sample_NolanM.py"

st.set_page_config(page_title="Stock Price Prediction LSTM", page_icon=":chart_with_upwards_trend:")
st.sidebar.header("Project NolanM")
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

st.markdown(
    """
        <style>
            .st-emotion-cache-13ln4jf.ea3mdgi5 {
                max-width: 1200px;
            }
        </style>
    """, unsafe_allow_html=True)

st.title("Stock Price Prediction with LSTM Model - NolanM")
st.write("")
st.markdown(
    """
    **Objective**: The goal of this project is to integrate big data workflows, including data extraction, analysis, and visualization, using historical stock price data from Yahoo Finance. This project focuses on developing a predictive model for stock prices, particularly for Tesla Inc. (TSLA), leveraging deep learning techniques. By exploring various aspects of stock price trends and validating the model's predictions, the project aims to provide insights into stock price forecasting and model performance evaluation.
    """
)

st.markdown("---")

st.markdown("""
    ## I. Initialization Variables
    -  Initialize the predictor with the stock symbol and date range.
    - **stock_symbol**: The stock symbol of the company you want to predict.
    - **start_date**: The start date for the historical stock data.
""")
code_Initialization = '''
    def __init__(self, stock_symbol, start_date):
        self.stock_symbol = stock_symbol
        self.start_date = start_date
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0,1))
    '''
st.code(code_Initialization, language='python')

st.markdown("---")

st.markdown("""
    ## Data Fetching
    - Historical stock data for the given symbol and date range is fetched using Yahoo Finance (`yfinance`).
    - The data is cleaned by removing the `Date` and `Adj Close` columns, which are not needed for this model.
""")

code_Fetching = '''
    def fetch_data(self):
        df = yf.download(self.stock_symbol, start=self.start_date, end=self.end_date)
        df_reset = df.reset_index()
        self.df_drop = df_reset.drop(['Date', 'Adj Close'], axis=1)
'''

st.code(code_Fetching, language='python')

st.markdown("---")

st.markdown("""
    ## Data Preparation
    - The data is split into training (70%), validating (20%), and testing (10%) datasets.
    - The `Close` prices are transformed using the `MinMaxScaler`.
    - Training data is prepared in a format suitable for LSTM: sequences of 100 previous data points (`x_train`) are used to predict the next data point (`y_train`). 
""")

code_Preparation = '''
    def prepare_data(self):
        index_training = int(len(self.df_drop) * 0.7)
        index_validating = int(len(self.df_drop) * 0.9)  # 70% + 20%
        index_testing = len(self.df_drop)  # The remaining 10%

        data_training = pd.DataFrame(self.df_drop['Close'][:index_training])
        data_validating = pd.DataFrame(self.df_drop['Close'][index_training:index_validating])
        data_testing = pd.DataFrame(self.df_drop['Close'][index_validating:index_testing])

        data_training_transform = self.scaler.fit_transform(data_training)

        x_train = []
        y_train = []

        for i in range(100, data_training_transform.shape[0]):
            x_train.append(data_training_transform[i-100 : i])
            y_train.append(data_training_transform[i, 0])

        self.x_train, self.y_train = np.array(x_train), np.array(y_train)
        self.data_validating = data_validating
        self.data_training = data_training
'''

st.code(code_Preparation, language='python')

st.markdown("---")

st.markdown("""
    ## Model Building
    - A Sequential model is built with four LSTM layers and Dropout layers to prevent overfitting.
    - The model is compiled using the Adam optimizer and Mean Squared Error loss function.

    **Note**: 
    - Sequential model is a linear stack of layers.
    - LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) architecture that is well-suited for time series data.
    - Dropout is a regularization technique that helps prevent overfitting by randomly setting a fraction of input units to zero during training.
    - Adam is an optimization algorithm that computes adaptive learning rates for each parameter.
    - Mean Squared Error is a loss function used in regression problems.

    **Model Architecture**:
    - LSTM (50 units) -> Dropout (0.2)
    - LSTM (60 units) -> Dropout (0.3)
    - LSTM (80 units) -> Dropout (0.4)
    - LSTM (150 units) -> Dropout (0.5)
    - Dense (1 unit) -> Output layer

    **LSTM (Long Short-Term Memory) Layer**:
        - Units: 50 neurons in this layer. Each neuron is capable of capturing long-term dependencies in the data.
        - Activation: 'relu' (Rectified Linear Unit), a common activation function that helps the network learn complex patterns.
        - Return Sequences: True, which means that this layer returns the full sequence of outputs for each input sample. This is necessary because we are stacking multiple LSTM layers.
        - Input Shape: (self.x_train.shape[1], 1), specifying the shape of the input data. self.x_train.shape[1] is the number of time steps, and 1 is the number of features.

""")

code_Model = '''
    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(units=60, activation='relu', return_sequences=True))
        self.model.add(Dropout(0.3))

        self.model.add(LSTM(units=80, activation='relu', return_sequences=True))
        self.model.add(Dropout(0.4))

        self.model.add(LSTM(units=150, activation='relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(units=1))  # 1 unit = Closing prices

        self.model.compile(optimizer='adam', loss='mean_squared_error')
'''

st.code(code_Model, language='python')

st.image(IMAGE_MODEL_SUMMARY, caption='Model Summary')

st.markdown("---")

st.markdown("""
    ## Model Training
    - The model is trained on the training data for a specified number of epochs (default is 50).
    - The Mean Squared Error loss function is used to optimize the model.
    - The trained model is saved as a Keras model file (`keras_model.keras`).
""")

code_Training = '''
    def train_model(self, epochs=50):
        self.model.fit(self.x_train, self.y_train, epochs=epochs)
        self.model.save('keras_model.keras')
'''

st.code(code_Training, language='python')

st.image(IMAGE_TRAINING_PROCESSING, caption='Training Process')

st.markdown("---")

st.markdown("""
    ## Validation Data Preparation
    - The validating data is prepared similarly to the training data, including past 100 days of training data to predict the validating data.
    - The data is scaled using the same MinMaxScaler used for training data.
""")

code_Validation = '''
    def validate_model(self):
        past_100_days_validating = self.data_training.tail(100)
        final_df_validating = pd.concat([past_100_days_validating, self.data_validating], ignore_index=True)

        input_validating_data = self.scaler.fit_transform(final_df_validating)

        x_validate = []
        y_validate = []

        for i in range(100, input_validating_data.shape[0]):
            x_validate.append(input_validating_data[i - 100 : i])
            y_validate.append(input_validating_data[i, 0])

        self.x_validate_np, self.y_validate_np = np.array(x_validate), np.array(y_validate)
'''

st.code(code_Validation, language='python')

st.markdown("---")

st.markdown("""
    ## Prediction
    - Predictions are made on the validating data using the trained model.
    - The predicted values and actual values are scaled back to their original scale using the inverse of the scaler factor.

    **Note**:
    - The scale factor is calculated as the inverse of the scale used by the MinMaxScaler.
    - You can adjust the scale factor based on the actual scale used in your data.
    - You can retrive thhe scaler factor by printing `self.scaler.scale_` in the `predict` method.
    - The predicted and actual values are returned for further analysis.

""")

code_Prediction = '''
    def predict(self):
        y_validate_predicted = self.model.predict(self.x_validate_np)
        print(self.scaler.scale_)
        scale_factor = 1/0.00252999
        y_predicted_scale_again = y_validate_predicted * scale_factor
        y_validate_scale_again = self.y_validate_np * scale_factor

        return y_predicted_scale_again, y_validate_scale_again
'''

st.code(code_Prediction, language='python')

st.markdown("---")

st.markdown("""
    ## Visualization
    - The line graphs shows the predicted stock prices against the actual stock prices.
    - The input ticker symbol is `TSLA` for Tesla Inc.
    - The start date is set to `2010-01-01`.
    - The model is trained for 50 epochs.

""")

st.image(IMAGE_RESULT, caption='Results')

st.markdown("---")

st.markdown("""
    ## Conclusion
    - The LSTM model is trained on historical stock price data to predict future stock prices.
    - The RMSE (Root Mean Squared Error) approximates around 0.00257015, indicating a good fit for the model with input TSLA stock data in 14 years.
""")

st.markdown("---")

st.write("")

st.markdown("""
    ## Download Files
    - You can download the Keras model file (`keras_model.keras`) and the sample script file (`Training_Model_Sample_NolanM.py`) below.
""")

with open(MODEL_SAVE_PATH, 'rb') as model_file:
    model_data = model_file.read()

with open(SAMPLE_FILE, 'r') as file:
    file_data = file.read()

# Create a download button for the .keras model file
st.download_button(
    label="Download Keras Model",
    data=model_data,
    file_name="LSTM_Stock_Model_TSLA_Trainned.keras",
    mime="application/octet-stream"
)

# Create a download button for the sample file
st.download_button(
    label="Download Sample Script File",
    data=file_data,
    file_name="Training_Model_Sample_NolanM.py",
    mime="text/plain"
)

st.markdown("---")



