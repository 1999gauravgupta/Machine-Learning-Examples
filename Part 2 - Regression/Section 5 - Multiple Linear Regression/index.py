import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("50_Startups.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x=LabelEncoder()
x[:,3]=labelencoder_x.fit_transform(x[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()

x=x[:,1:]

import statsmodels.formula.api as sm
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
x_opt=x[:,[0,1,2,3,4,5]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
x_opt=x[:,[0,1,3,4,5]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
x_opt=x[:,[0,3,4,5]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
x_opt=x[:,[0,3,5]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
x_opt=x[:,[0,3]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_opt,y,test_size=0.2,random_state=0)

x_train=x_train[:,[1]]
x_test=x_test[:,[1]]

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_train_pred=regressor.predict(x_train)

plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,y_train_pred,color="blue")
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

plt.scatter(x_test,y_test,color="red")
plt.plot(x_train,y_train_pred,color="blue")
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()