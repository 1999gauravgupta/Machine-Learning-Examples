#FIXXXXXXXXXXXXX!
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

dataset=pd.read_csv("Salary_Data.csv")

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.preprocessing import StandardScaler
scaler_x=StandardScaler()
x_train=scaler_x.fit_transform(x_train)

#Linear Regressor model building
#Model Shape:y=a+bx
#For b
num=0
den=0
num1=x_train-x_train.mean()
num2=y_train-y_train.mean()
num=(num1*num2).sum()
den=((x_train-x_train.mean())**2).sum()
b=num/den
a=y_train.mean()-(b*x_train.mean())
arr=[]
for i in range(len(x_test)):
    arr.append(float(a+b*(x_test[i])))
    
##my model done now we will compare with module model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

plt.scatter(x_test,y_test,color="green")
plt.plot(x_test,y_pred,color="red")
plt.plot(x_test,arr,color="blue")
