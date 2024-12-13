import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path="C://Users//ACER//Desktop//Lab//9//WEEK9.xlsx"
sheets=["SET1","SET2"]
for sheet in sheets:
    df=pd.read_excel(path, sheet)
    S=(df["X"]).to_numpy()
    T=(df["T"]).to_numpy()
    w=S*0.0314
    maxT=np.max(T)
    w_avg=np.average(w)
    w_avg_around=np.around(w_avg, 3)
    abserror=0
    for i in range(0,len(T)):
        abs_error=np.abs(w[i]-w_avg)
    MAPE=np.around((np.abs(w[i]-w_avg)/w_avg*100), 3)

    plt.title(f"ω Versus T Graph for {sheet}")
    plt.plot([T[0],T[-1]],[w_avg, w_avg], color="r")
    plt.scatter(T, w, marker=".", color="b", s=50)
    plt.xlim(left=0, right=round(maxT)+1)
    plt.ylim(bottom=0, top=np.max(w)+1)
    plt.xticks(np.arange(0,round(maxT)+1,1))
    plt.yticks(np.arange(0,np.max(w)+1,1))
    plt.xlabel("Time(seconds)")
    plt.ylabel("ω (rad/s)")
    plt.legend([f"Average ω value:{w_avg_around}",f"Experimental ω value \nMean Absolute Percentage Error:{MAPE}%"], loc="lower right", facecolor="w")
    plt.show()