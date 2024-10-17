import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

fpath="C://Users//ACER//Desktop//Lab//exp3.xlsx"
sheets=["10hz 1st","20hz 1st","10hz 2nd","20hz 2nd","10hz 3rd","20hz 3rd","10hz 4th", "20hz 4th"]

for sheet in sheets:
    df=pd.read_excel(fpath, sheet_name=sheet)
    X = df['X'].to_numpy().reshape(-1, 1)  
    Y = df['Y'].to_numpy().reshape(-1, 1) 
    model= LinearRegression(fit_intercept=False)
    model.fit(X,Y)
    y_predict=model.predict(X)

    plt.scatter(X, Y, color = 'firebrick', s=8)
    plt.plot(X, y_predict, color = 'steelblue')
    plt.xticks(df['X'])
    plt.yticks(range(int(Y.min()), int(np.ceil(Y.max()))+ 1)) 
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel('Time(s)')
    plt.ylabel('X(cm)')
    plt.legend(["Data", 'Best fit line'], title = f'total displacement versus time for {sheet} Set', loc='best', facecolor='whitesmoke')
    plt.show()