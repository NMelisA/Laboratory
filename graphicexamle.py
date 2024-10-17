import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df=pd.read_excel("C://Users//ACER//Desktop//dataset.xlsx" )
X = df['V'].to_numpy().reshape(-1, 1)  
Y = df['m'].to_numpy().reshape(-1, 1) 
model= LinearRegression()
model.fit(X,Y)
y_predict=model.predict(X)

plt.scatter(X, Y, color = 'lightcoral')
plt.plot(X, y_predict, color = 'firebrick')
plt.xlabel('Mass')
plt.ylabel('Volume')
plt.legend(["best fit line", 'data'], title = 'mass/Volume', loc='best', facecolor='white')
plt.show()