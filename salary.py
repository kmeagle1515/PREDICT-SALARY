import pandas as pd
import csv
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model

#data_frame = pd.read_csv('Salary_Data.csv', names=['in', 'out'],

# index_col=1, parse_dates=True)

#data_frame.head()

#data_ask =  data_frame['Ask'].resample('5Min').ohlc()

#data_bid =  data_frame['Bid'].resample('5Min').ohlc()

#export_csv=data_ask.to_csv(r'C:\Users\Owner\Desktop\new.csv',index=None,header=True)

#data_bid.head()

#data_ask_bid=pd.concat([data_ask, data_bid], axis=1, keys=['Ask', 'Bid'])
data_frame = pd.read_csv('Salary_Data.csv')

#
x_train=data_frame.iloc[2:23].values
y_train=data_frame.iloc[2:23,1].values
x_test=data_frame.iloc[24:31].values
y_test=data_frame.iloc[24:31,1].values





ols = linear_model.LinearRegression()
#ols=SVR(kernel='linear',degree=1)
ols.fit(x_train, y_train)
pred=ols.predict(x_test)


#print(r2_score(y_test,pred))
print(r2_score(y_test,pred))
print (pred)