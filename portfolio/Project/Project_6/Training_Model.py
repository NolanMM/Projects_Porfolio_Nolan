from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class StockPricePredictor:
    def __init__(self, stock_symbol, start_date):
        """
        Initialize the predictor with the stock symbol and date range.
        """
        self.stock_symbol = stock_symbol
        self.start_date = start_date
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0,1))

    def fetch_data(self):
        """
        Fetch historical stock data from Yahoo Finance.
        """
        df = yf.download(self.stock_symbol, start=self.start_date, end=self.end_date)
        df_reset = df.reset_index()
        self.df_drop = df_reset.drop(['Date', 'Adj Close'], axis=1)

    def prepare_data(self):
        """
        Prepare training, validating, and testing datasets.
        """
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

    def build_model(self):
        """
        Build the LSTM neural network model.
        """
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

    def train_model(self, epochs=50):
        """
        Train the model with the training data.
        """
        self.model.fit(self.x_train, self.y_train, epochs=epochs)
        self.model.save('keras_model.keras')

    def validate_model(self):
        """
        Prepare the validating dataset.
        """
        past_100_days_validating = self.data_training.tail(100)
        final_df_validating = pd.concat([past_100_days_validating, self.data_validating], ignore_index=True)
        
        # Ensure column names are strings
        #final_df_validating.columns = final_df_validating.columns.astype(str)
        
        input_validating_data = self.scaler.fit_transform(final_df_validating)

        x_validate = []
        y_validate = []

        for i in range(100, input_validating_data.shape[0]):
            x_validate.append(input_validating_data[i - 100 : i])
            y_validate.append(input_validating_data[i, 0])

        self.x_validate_np, self.y_validate_np = np.array(x_validate), np.array(y_validate)

    def predict(self):
        """
        Predict stock prices using the trained model.
        """
        y_validate_predicted = self.model.predict(self.x_validate_np)
        print(self.scaler.scale_)
        scale_factor = 1/0.00252999
        y_predicted_scale_again = y_validate_predicted * scale_factor
        y_validate_scale_again = self.y_validate_np * scale_factor

        return y_predicted_scale_again, y_validate_scale_again

    # def plot_predictions(self, y_predicted, y_actual):
    #     """
    #     Plot the predicted stock prices against the actual prices.
    #     """
    #     plt.figure(figsize=(14, 5))
    #     plt.plot(y_actual, color='blue', label='Actual Stock Price')
    #     plt.plot(y_predicted, color='red', label='Predicted Stock Price')
    #     plt.title('Stock Price Prediction')
    #     plt.xlabel('Time')
    #     plt.ylabel('Stock Price')
    #     plt.legend()
    #     plt.show()

    # def calculate_mse(self, y_predicted, y_actual):
    #     """
    #     Calculate and print the Mean Squared Error (MSE) between the predicted and actual prices.
    #     """
    #     mse = mean_squared_error(y_actual, y_predicted)
    #     print(f'Mean Squared Error: {mse}')

if __name__ == "__main__":
    predictor = StockPricePredictor('TSLA', '2010-01-01')
    predictor.fetch_data()
    predictor.prepare_data()
    predictor.build_model()
    predictor.train_model()
    predictor.validate_model()
    y_predicted, y_actual = predictor.predict()
    #predictor.plot_predictions(y_predicted, y_actual)
    #predictor.calculate_mse(y_predicted, y_actual)
