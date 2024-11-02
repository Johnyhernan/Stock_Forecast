import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow as tf
import math
import datetime
from sklearn.metrics import mean_squared_error
#what we would use to calculate percentage 
from sklearn.preprocessing import MinMaxScaler
# we will use this to scale our data making it more readable and allow efficent results for our LSTM models


training_data=pd.read_csv('appl_data.csv')
#load the following csv file using pandas
training_data=training_data.iloc[::-1].reset_index(drop=True)
#^ needed to reverse csv file to properly portray it
training_data['Close/Last'] = training_data['Close/Last'].str.replace('$', '', regex=False)  
training_data['Close/Last'] = training_data['Close/Last'].str.replace(',', '', regex=False)
training_data['Close/Last'] = pd.to_numeric(training_data['Close/Last'])
# train data is an object we will use the following to covert it into numerical data, which will be float62
training_data['Date'] = pd.to_datetime(training_data['Date'])

train_temp=training_data.reset_index()['Close/Last']
#make a train_temp that will only contain the closing price, this will be the data we will use for our lstm models
scaler=MinMaxScaler(feature_range=(0,1))
#making a scaler
train_temp=scaler.fit_transform(np.array(train_temp).reshape(-1,1))
# train_temp is turned into a numpy array then scaled 
training_size=int(len(train_temp)*0.80)
test_size=len(train_temp)-training_size
#splitting our data into 20% testing and 80%  training
train_data,test_data=train_temp[0:training_size,:],train_temp[training_size:len(train_temp),:1]
#-------
def create_dataset(dataset, time_step=1):
    # Ensure the data is in a two-dimensional format for consistent indexing
    if isinstance(dataset, pd.Series):
        dataset = dataset.to_frame()
    # Access the first (and only) column by position
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i+time_step), 0] 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)



time_step = 100
#since we are working with a large dataset we will use a time step of 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
#Creating model we will use for testing with 3 layers
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=200,batch_size=64,verbose=1)
#we will use a default number of epochs which is 100 with a batch size of 64
# A verbose of 1 indicates it will show the model testing
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
# we will use numpy to create a matrix



train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Transform back to original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Calculate RMSE 
train_rmse = math.sqrt(mean_squared_error(y_train, train_predict))
test_rmse = math.sqrt(mean_squared_error(ytest, test_predict))
print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)

predicted_prices=[]
n_steps=100
i=0
#index should be zero
x_input=test_data[152:].reshape(1,-1)

temp_input=list(x_input)
temp_input=temp_input[0].tolist()

Number_of_days=5
#number of days you want to forecast


while(i<Number_of_days):
    
    if(len(temp_input)>100):
        x_input=np.array(temp_input[1:])
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        predicted_prices.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        predicted_prices.extend(yhat.tolist())
        i=i+1

predicted_prices=scaler.inverse_transform(predicted_prices)
Mixed_train=scaler.inverse_transform(train_temp).tolist()
Mixed_train.extend(predicted_prices)


#adding days to numpy array so that we can see predicted data
temp=np.array(training_data['Date'])
new_dates = pd.date_range(start='2024-10-31', periods=len(predicted_prices), freq='D').to_numpy()
temp = np.concatenate((temp, new_dates))
#turning mixed_train into dataframe with columns since its a numpy array at the moment
df = pd.DataFrame(Mixed_train,columns=['CLOSE'])
df['DATE']=pd.DataFrame(temp)



#calculating moving averages for the chart using pandas rolling window
df['Moving_Average-100'] = df['CLOSE'].rolling(window=100).mean()
df['Moving_Average-20'] = df['CLOSE'].rolling(window=20).mean()
#making a new column in df that will include the date format for most values
x=df['DATE']
#ploting data
plt.plot(x,df['CLOSE'],label='Close Price',color='black')
plt.plot(x,df['Moving_Average-100'],label='Moving Average- 100 day',color='orange')
plt.plot(x,df['Moving_Average-20'],label='Moving Average- 20 day',color='green')
plt.legend()
plt.title('AAPL Price Prediction for Next '+str(len(predicted_prices))+' days')
plt.xlabel('Days')
plt.ylabel('Price')
plt.grid(True)
plt.show()


profit=df.iloc[-1]['CLOSE']-df.iloc[-(len(predicted_prices)+1)]['CLOSE']
print ('profit: '+str(profit))











