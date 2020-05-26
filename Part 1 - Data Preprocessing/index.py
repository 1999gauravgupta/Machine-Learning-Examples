#importing basic libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#opening data file
dataset=pd.read_csv("Data.csv")

#x is list of what we know and y is what we want to predict
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

#this is to fill missing value with avg of that column
from sklearn.preprocessing import Imputer
imputer=Imputer()
imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])

#encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x=LabelEncoder()
x[:,0]=labelencoder_x.fit_transform(x[:,0])
#labels here are considered as order by algorithms to remove that we use this
onehotencoder=OneHotEncoder(categorical_features=[0])
x=onehotencoder.fit_transform(x).toarray()
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

#splitting dataset into training and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#feature scaling cuz large number variation increases computatuional resources
from sklearn.preprocessing import StandardScaler
scaler_x=StandardScaler()
x_train=scaler_x.fit_transform(x_train)
x_test=scaler_x.transform(x_test)