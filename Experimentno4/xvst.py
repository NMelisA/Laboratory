import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

fpath="C://Users//ACER//Desktop//Lab//WEEK4.xlsx"
sheets=["1st","2nd","3rd","4th","5th"]

#x vs t
for sheet in sheets:
    df=pd.read_excel(fpath, sheet_name=sheet)
    df_10 = df[['X1', 'Y1']].dropna()  
    df_15 = df[['X2', 'Y2']].dropna()

    #for 10 degree slope
    X1= df_10['X1'].to_numpy().reshape(-1, 1)
    Y1= df_10['Y1'].to_numpy().reshape(-1, 1) 
    poly=PolynomialFeatures(degree=2, include_bias=False)
    poly_features1=poly.fit_transform(X1)
    model1= LinearRegression(fit_intercept=False)
    model1.fit(poly_features1,Y1)
    y_predict1=model1.predict(poly_features1)
    coef1 = np.around( model1.coef_.flatten(),2)
    intercept1 = np.around(model1.intercept_ ,2)
    eq1=f"({coef1[0]})x²+({coef1[1]})x+{intercept1}"

    #for 15 degree slope

    X2= df_15['X2'].to_numpy().reshape(-1, 1)  
    Y2= df_15['Y2'].to_numpy().reshape(-1, 1) 
    poly= PolynomialFeatures(degree=2, include_bias=False)
    poly_features2= poly.fit_transform(X2)
    model2= LinearRegression(fit_intercept=False)
    model2.fit(poly_features2,Y2)
    y_predict2= model2.predict(poly_features2)
    coef2= np.around( model2.coef_.flatten(),2)
    intercept2= np.around(model2.intercept_ ,2)
    eq2=f"({coef2[0]})x²+({coef2[1]})x+{intercept2}"

    x_ticks = np.unique(np.concatenate((Y1.flatten(), Y2.flatten())))
    x_max = max(Y1.max(), Y2.max())
    y_max = max(X1.max(), X2.max()) 
    #10 degree
    plt.scatter(Y1, X1, color = 'firebrick', s=20, marker=".")
    plt.plot(y_predict1, X1, color = 'tomato')
    
    #15 degree
    plt.scatter(Y2, X2, color = 'navy', s=20, marker=".")
    plt.plot(y_predict2, X2, color = 'steelblue')

    plt.xlim(left=0, right=x_max + 0.05)
    plt.ylim(bottom=0, top=y_max + 1)
    plt.xticks(x_ticks)
    plt.yticks(np.arange(0, y_max + 1, 1))
    plt.xlabel('Time(s)')
    plt.ylabel('X(cm)')
    plt.title(f"Polynomial Fit for Distance vs Time at 10° and 15° Slopes {sheet} Set")
    plt.legend(["Data at 10°", f'Polynomial Fit={eq1}',"Data at 15°", f'Polynomial Fit={eq2}'], loc='best', facecolor='w')
    plt.show()