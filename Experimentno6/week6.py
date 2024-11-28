import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

path= "C://Users//ACER//Desktop//Lab//6//WEEK6.xlsx"
sheets=["1and0","2and0","2and1"]
for sheet in sheets:
    i=1 #steps for plotting
    df=pd.read_excel(path, sheet_name=sheet)
    M1=df.columns[0]#header names
    M2=df.columns[1]
    Data= df[[M1,M2,"t"]].dropna()  
    X = Data[M1].to_numpy().reshape(-1, 1) #The greatest mass for the dataset
    Y = Data[M2].to_numpy().reshape(-1, 1) #The elapsed time
    T = Data['t'].to_numpy().reshape(-1, 1) #The lightest mass for the dataset

    #to fit a polynomial for x&y vs t graphs
    poly= PolynomialFeatures(degree=2, include_bias=False)
    poly_features= poly.fit_transform(T)

    #model for x-t (greater mass)
    x_model= LinearRegression()
    x_model.fit(poly_features, X)
    x_predict= x_model.predict(poly_features)
    x_coef= np.around( x_model.coef_.flatten(),1)
    x_intercept= np.around(x_model.intercept_ ,1)
    x_eq=f"({x_coef[0]})y²+({x_coef[1]})y+{x_intercept}"
    x_mse= np.around(mean_squared_error(X, x_predict),1)

    #model for y-t (lighter mass)
    y_model= LinearRegression()
    y_model.fit(poly_features, Y)
    y_predict= y_model.predict(poly_features)
    y_coef= np.around( y_model.coef_.flatten(),1)
    y_intercept= np.around(y_model.intercept_ ,1)
    y_eq=f"({y_coef[0]})y²+({y_coef[1]})y+{y_intercept}"
    y_mse= np.around(mean_squared_error(Y, y_predict),1)

    T2=np.square(T)
    #Model for x&y vs t² graph
    x2_model= LinearRegression()
    x2_model.fit(T2, X)
    x2_predict=x2_model.predict(T2)
    x2_slope= np.around(x2_model.coef_.item(),1)
    x2_mse= np.around(mean_squared_error(X, x2_predict),1)

    y2_model= LinearRegression()
    y2_model.fit(T2, Y)
    y2_predict=y2_model.predict(T2)
    y2_slope= np.around(y2_model.coef_.item(),1)
    y2_mse= np.around(mean_squared_error(Y, y2_predict),1)

    #finding min and max values for plotting     
    max=np.max(X)
    if max < np.max(Y):
        max= np.max(Y)
    min=np.min(X)
    if np.min(Y) < min:
        min= np.min(Y)

    if i==1: #plotting fot x&y vs t graph
        plt.scatter(T, X, color = 'navy', s=20, marker=".")
        plt.plot(T, x_predict,  color = 'steelblue')

        plt.scatter(T, Y, color = 'firebrick', s=20, marker=".")
        plt.plot(T, y_predict,  color = 'lightcoral')

        plt.xlim(left=0, right=np.max(T) + 0.05)
        plt.ylim(bottom=min-1, top=max+1)
        plt.xlabel('Time(s)')
        plt.ylabel('Y(cm)')
        plt.xticks(np.arange(0, np.max(T)+0.1, 0.1))
        plt.yticks(np.arange(min, max+1, 1))
        plt.title(f"Polynomial Fit for Distance vs time for the {M1} vs {M2}")
        plt.legend([f"Experimental Data for {M1}", 
                    f"Polynomial Fit={x_eq}\nMean squared error={x_mse}",f"Experimental Data for {M2}",
                    f"Polynomial Fit={y_eq} \nMean squared error={y_mse}"], loc='best', facecolor='w')
        plt.show()
        i=2
    
    if i==2:
        plt.scatter(T2, X, color = 'navy', s=20, marker=".")
        plt.plot( T2, x2_predict , color = 'steelblue')

        plt.scatter(T2, Y , color = 'firebrick', s=20, marker=".")
        plt.plot(T2, y2_predict, color = 'lightcoral')

        plt.xlim(left=0, right=np.max(T2) + 0.1)
        plt.ylim(bottom=min-1, top=max+1)
        plt.xticks(T2.flatten())
        plt.yticks(np.arange(min, max+1, 1))
        plt.xlabel('Time(s)')
        plt.ylabel('Y(cm)')
        plt.title(f"linear Fit for Distance vs t for the {M1} vs {M2}")
        plt.legend([f"Experimental Data for {M1}",f"Linear Fit={x2_slope}x\nMean squared error={x2_mse}"
                    ,f"Experimental Data for {M2}",f"Linear Fit={y2_slope}x \nMean squared error={y2_mse}"]
                    , loc='best', facecolor='w')
        plt.show()
        i=1