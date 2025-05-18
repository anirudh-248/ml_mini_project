import streamlit as st
import datetime
from pandas_datareader import data as pdr
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers # type: ignore
from tensorflow.keras import datasets, layers, models # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM,Dropout # type: ignore
from sklearn.metrics import accuracy_score
from tensorflow.keras import datasets, layers,models # type: ignore
from tensorflow.keras.models import Model, load_model # type: ignore
from tensorflow.keras.layers import Input, Dense, Activation,SimpleRNN # type: ignore
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
tf.random.set_seed(7)

st.title('Stock Prediction using Deep Learning')
st.subheader('Select the method of input:')
option = st.radio('Radio', ["Upload the data (.csv format)","Get data from the net"])

class stock_predict_DL:
    
    def __init__(self,comp_df):
        data = comp_df.filter(['Open'])
        dataset = data.values
        st.subheader('How much percent of the data needs to be allocated for training?')
        st.text('Default is set to 90')
        perc_train = st.number_input('',step = 1,min_value=1, value = 90)
        training_data_len = int(np.ceil( len(dataset) * (perc_train/100)))
        self.scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = self.scaler.fit_transform(dataset)
        train_data = scaled_data[0:int(training_data_len), :]
        self.X_train = []
        self.y_train = []
        
        for i in range(k, len(train_data)):
            self.X_train.append(train_data[i-k:i, 0])
            self.y_train.append(train_data[i, 0])

        self.X_train, self.y_train = np.array(self.X_train), np.array(self.y_train)

        test_data = scaled_data[training_data_len - k: , :]
        self.X_test = []
        self.y_test = dataset[training_data_len:, :]
        for i in range(k, len(test_data)):
            self.X_test.append(test_data[i-k:i, 0])

        self.X_test = np.array(self.X_test)
        test_dates = comp_df['Date'].values
        self.testd = test_dates[training_data_len:] 
        
    def LSTM_model(self):
        
        st.title("Long Short-Term Memory (LSTM)")
        Xtrain = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        Xtest = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1 ))
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape= (Xtrain.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(32))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(Xtrain, self.y_train, batch_size=1, epochs= 1)
        predictions = model.predict(Xtest)
        predictions = self.scaler.inverse_transform(predictions)
        st.text("R2 SCORE")
        st.text(metrics.r2_score(self.y_test, predictions))
        st.text("MSLE")
        st.text(metrics.mean_squared_log_error(self.y_test, predictions))
        plt.plot(predictions)
        plt.plot(self.y_test)
        plt.legend(["Predicted","Observed"])
        step = max(1, len(self.testd) // 10)
        plt.xticks(range(0, len(self.testd), step), self.testd[::step], rotation=45)
        plt.xlabel('Date',fontsize=18)
        plt.ylabel('Price',fontsize=18)
        plt.title("LSTM")
        st.pyplot(plt)
        
    def Mlp_model(self):
        
        st.title("Multilayer perceptron (MLP)")
        regr = MLPRegressor(hidden_layer_sizes = 100, alpha = 0.01,solver = 'lbfgs',shuffle=True)
        regr.fit(self.X_train, self.y_train)
        y_pred = regr.predict(self.X_test)
        y_pred = y_pred.reshape(len(y_pred),1)
        y_pred = self.scaler.inverse_transform(y_pred)
        st.text("R2 SCORE")
        st.text(metrics.r2_score(self.y_test, y_pred))
        st.text("MSLE")
        st.text(metrics.mean_squared_log_error(self.y_test, y_pred))
        plt.plot(y_pred)
        plt.plot(self.y_test)
        plt.legend(["Predicted","Observed"])
        step = max(1, len(self.testd) // 10)
        plt.xticks(range(0, len(self.testd), step), self.testd[::step], rotation=45)
        plt.xlabel('Date',fontsize=18)
        plt.ylabel('Price',fontsize=18)
        plt.title("MLP")
        st.pyplot(plt)
        
    def basic_ann_model(self):
        
        st.title("Basic Artificial Neural Network (ANN)")
        classifier = Sequential()
        classifier.add(Dense(units = 128, activation = 'relu', input_dim = self.X_train.shape[1]))
        classifier.add(Dropout(0.2))
        classifier.add(Dense(units = 64))
        classifier.add(Dropout(0.2))
        classifier.add(Dense(units = 1))
        classifier.compile(optimizer = 'adam', loss = 'mean_squared_error')
        classifier.fit(self.X_train, self.y_train, batch_size = 32, epochs = 10)
        prediction = classifier.predict(self.X_test)
        y_pred = self.scaler.inverse_transform(prediction)
        st.text("R2 SCORE")
        st.text(metrics.r2_score(self.y_test, y_pred))
        st.text("MSLE")
        st.text(metrics.mean_squared_log_error(self.y_test, y_pred))
        plt.plot(y_pred)
        plt.plot(self.y_test)
        plt.legend(["Predicted","Observed"])
        step = max(1, len(self.testd) // 10)
        plt.xticks(range(0, len(self.testd), step), self.testd[::step], rotation=45)
        plt.xlabel('Date',fontsize=18)
        plt.ylabel('Price',fontsize=18)
        plt.title("ANN")
        st.pyplot(plt)
    
    def rnn_model(self):
        
        st.title("Recurrent neural network (RNN)")
        Xtrain = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        Xtest = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1 ))
        model = Sequential()
        model.add(SimpleRNN(units=4, input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(Xtrain, self.y_train, epochs=10, batch_size=1)
        prediction = model.predict(Xtest)
        y_pred = self.scaler.inverse_transform(prediction)
        st.text("R2 SCORE")
        st.text(metrics.r2_score(self.y_test, y_pred))
        st.text("MSLE")
        st.text(metrics.mean_squared_log_error(self.y_test, y_pred))
        plt.plot(y_pred)
        plt.plot(self.y_test)
        plt.legend(["Predicted","Observed"])
        step = max(1, len(self.testd) // 10)
        plt.xticks(range(0, len(self.testd), step), self.testd[::step], rotation=45)
        plt.xlabel('Date',fontsize=18)
        plt.ylabel('Price',fontsize=18)
        plt.title("RNN")
        st.pyplot(plt)

flag = "False"

if option == "Get data from the net":
    st.sidebar.subheader('Query parameters')
    start_date = st.sidebar.date_input("Start date", datetime.date(2012, 5, 18))
    end_date = st.sidebar.date_input("End date", datetime.date(2021,3, 25))
    ticker_list = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt')
    tickerSymbol = st.sidebar.selectbox('Stock ticker', ticker_list) 
    data = pdr.get_data_yahoo(tickerSymbol, start = start_date, end = end_date).reset_index()
    data['Date'] = pd.to_datetime(data['Date']).dt.date
    flag = "True"
    st.header('**Stock data**')
    st.write(data)

elif option == "Upload the data (.csv format)":
    file = st.file_uploader('Dataset')
    if file is not None:
        data = pd.read_csv(file)
        flag = "True"
        st.header('**Stock data**')
        st.write(data)

if flag == "True":
    st.subheader('Define time window length:')
    st.text('Default is set to 60')
    k = st.number_input('',step = 1,min_value=1, value = 60)
    company_stock = stock_predict_DL(data)
    st.subheader('Which Deep Learning model would you like to train? :')
    mopt = st.selectbox('', ["Click to select", "LSTM","MLP","RNN","Basic ANN","Autoencoder"])

    if mopt=="LSTM":
        company_stock.LSTM_model()

    if mopt=="MLP":
        company_stock.Mlp_model()

    if mopt == "RNN":
        company_stock.rnn_model()

    if mopt == "Basic ANN":
        company_stock.basic_ann_model()
