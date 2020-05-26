import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("Position_Salaries.csv")

x=dataset.iloc[:,[1]].values
y=dataset.iloc[:,[-1]].values

from sklearn.tree import DecisionTreeRegressor

regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

x_grid=np.arange(min(x),max(x),0.01)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color="red")
plt.plot(x_grid,regressor.predict(x_grid),color="blue")
plt.title("decision tree regression")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()