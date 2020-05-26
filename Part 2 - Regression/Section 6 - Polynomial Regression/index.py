import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("Position_Salaries.csv")

x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)
regressor=LinearRegression()
regressor.fit(x_poly,y)

x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape((len(x_grid),1))

plt.scatter(x,y,color="red")
plt.plot(x_grid,regressor.predict(poly_reg.fit_transform(x_grid)),color="blue")
plt.title("Salary vs Level")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()