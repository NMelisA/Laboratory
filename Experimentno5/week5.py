import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


fpath="C://Users//ACER//Desktop//Lab//5//WEEK5.xlsx"
sheets=["1st","2nd","3rd","4th","5th"]

for sheet in sheets:
    i=0#steps for plotting
    df=pd.read_excel(fpath, sheet_name=sheet)
    Data= df[['x',"t","y"]].dropna()  
    X = Data['x'].to_numpy().reshape(-1, 1)  
    T = Data['t'].to_numpy().reshape(-1, 1) 
    Y = Data['y'].to_numpy().reshape(-1, 1)

    #XT GRAPH
    modelx= LinearRegression(fit_intercept=False)
    modelx.fit(T,X)
    x_predict=modelx.predict(T)
    slope= np.around(modelx.coef_.item(),3)
    mse_X = np.around(mean_squared_error(X, x_predict),3)

    #YT GRAPH
    poly= PolynomialFeatures(degree=2, include_bias=False)
    poly_features= poly.fit_transform(T)
    modelY= LinearRegression(fit_intercept=False)
    modelY.fit(poly_features,Y)
    y_predict= modelY.predict(poly_features)
    coef= np.around( modelY.coef_.flatten(),3)
    intercept= np.around(modelY.intercept_ ,3)
    eq=f"({coef[0]})x²+({coef[1]})x+{intercept}"
    mse_Y = np.around(mean_squared_error(Y, y_predict),3)

    #YT2 GRAPH
    modely2= LinearRegression(fit_intercept=False)
    T2=np.square(T)
    modely2.fit(T2,Y)
    y_predict2=modely2.predict(T2)
    slope2= np.around(modely2.coef_.item(),3)
    mse_Y2 = np.around(mean_squared_error(Y, y_predict2),3)

    #XY GRAPH
    Y_calculated=[]
    for z in range(len(T)):
        t2=T[z]**2
        a=(np.sin(np.radians(15)))*980
        Y_cal=((a)*(t2)/2)
        Y_calculated.append(Y_cal)
    Y_calculated = np.array(Y_calculated)

    if i==0:#XT GRAPH
        plt.scatter(T, X, color = 'firebrick',s=20, marker="x")
        plt.plot(T,x_predict, color = 'steelblue')
        plt.xlabel('Time')
        plt.ylabel('Displacement')
        plt.xlim(left=0, right=np.max(T) + 0.05)
        plt.ylim(bottom=0, top=np.max(X) + 1)
        plt.xticks(np.arange(0, np.max(T)+0.05, 0.05))
        plt.yticks(np.arange(0, np.max(X)+1, 1))
        plt.title(f"Linear Fit for Displacement at x axis vs time {sheet} Set")
        plt.legend([f'Linear Fit={slope}x \nMean squared error={mse_X}','Experimental Data'], loc='best', facecolor='w')   
        plt.show()
        i=1
    if i==1:#YT GRAPH
        plt.scatter(T, Y, color = 'firebrick',s=20, marker="x")
        plt.plot(T,y_predict, color = 'steelblue')
        plt.xlabel('Time')
        plt.ylabel('Displacement')
        plt.xlim(left=0, right=np.max(T) + 0.05)
        plt.ylim(bottom=0, top=np.max(Y) + 1)
        plt.xticks(np.arange(0, np.max(T)+0.05, 0.05))
        plt.yticks(np.arange(0, np.max(Y)+1, 1))
        plt.title(f"Polynomial Fit for Displacement at y axis vs time {sheet} Set")
        plt.legend([ f'Polynomial Fit={eq} \nMean squared error={mse_Y}',"Experimental Data"], loc='best', facecolor='w')     
        plt.show()  
        i=2
    if i==2:#YT2 GRAPH
        plt.scatter(T2, Y, color = 'firebrick',s=20, marker="x")
        plt.plot(T2,y_predict2, color = 'steelblue')
        plt.xlabel('Time Squared')
        plt.ylabel('Displacement')
        plt.xlim(left=0, right=np.max(T2) + 0.05)
        plt.ylim(bottom=0, top=np.max(Y) + 1)
        plt.xticks(np.arange(0, np.max(T2)+0.05, 0.05))
        plt.yticks(np.arange(0, np.max(Y)+1, 1))
        plt.title(f"Linear Fit for Displacement at y axis vs time squared {sheet} Set")
        plt.legend([ f'Linear Fit={slope2} \nMean squared error={mse_Y2}',"Experimental Data"], loc='best', facecolor='w')  
        plt.show() 
        i=3   
    if i==3:#XY GRAPH 
        plt.scatter(X, Y, color = 'firebrick',s=20, marker="x")
        plt.scatter(X, Y_calculated, color = 'steelblue',s=40, marker=".")
        plt.gca().invert_yaxis()
        plt.xlabel('Location at X axis')
        plt.ylabel('Location at Y axis')
        plt.xlim(left=0, right=np.max(X) + 0.05)
        plt.ylim(bottom=np.max(Y) + 1, top=0)
        plt.xticks(np.arange(0, np.max(X)+1, 1))
        plt.yticks(np.arange(0, np.max(Y)+1, 1))
        plt.title(f"Expected pathway of motion vs experimental results for the {sheet} Set")
        plt.legend(["Experimental Data","Expected Data"], loc='best', facecolor='w')  
        plt.show() 
        i=0