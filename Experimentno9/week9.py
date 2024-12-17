import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path="C://Users//ACER//Desktop//Lab//9//WEEK9.xlsx"
sheets=["SET1"]

def MAPE(w, w_avg, T):
    for i in range(0,len(T)):
        MAPEValue=np.around((np.abs(w[i]-w_avg)/w_avg*100), 3)
    return MAPEValue

for sheet in sheets:
    df=pd.read_excel(path, sheet)
    S1=(df["S1"]).to_numpy()
    T1=(df["T1"]).to_numpy()
    w1=S1*0.0314
    w1_avg=np.average(w1)
    w1_avg_around=np.around(w1_avg, 3)
    MAPE1=MAPE(w1,w1_avg, T1)
    S2=(df["S2"]).to_numpy()
    T2=(df["T2"]).to_numpy()
    w2=S2*0.0314
    w2_avg=np.average(w2)
    w2_avg_around=np.around(w2_avg, 3)
    MAPE2=MAPE(w2,w2_avg, T2)

    maxT=np.max(T1)
    if np.max(T1)<np.max(T2):
        maxT=np.max(T2)  
    maxw=np.max(w1)
    if np.max(w1)<np.max(w2):
        maxw=np.max(w2)

    plt.title(f"ω Versus T Graph for {sheet}")
    plt.plot([T1[0],T1[-1]],[w1_avg, w1_avg], color="lightcoral")
    plt.plot([T2[0],T2[-1]],[w2_avg, w2_avg], color="steelblue")
    plt.scatter(T1, w1, marker=".", color="r", s=50)
    plt.scatter(T2, w2, marker=".", color="b", s=50)
    plt.xlim(left=0, right=round(maxT)+1)
    plt.ylim(bottom=0, top=maxw +1)
    plt.xticks(np.arange(0,round(maxT)+1,1))
    plt.yticks(np.arange(0,maxw +1,1))
    plt.xlabel("Time(seconds)")
    plt.ylabel("ω (rad/s)")
    plt.legend([f"Average ω value:{w1_avg_around}",f"Average ω value:{w2_avg_around}",
                f"Experimental ω value \nMean Absolute Percentage Error:{MAPE1}%",
                f"Experimental ω value \nMean Absolute Percentage Error:{MAPE2}%"], loc="lower right", facecolor="w")
    plt.show()
    print(f"For the 1st sub-set of {sheet} \n",
          f"Average S :{np.average(S1)}#lines/s \n",
          f"Average ω :{w1_avg_around}rad/s \n",
          f"Average ω :{(w1_avg_around)/(2*np.pi)}rev/s \n",
          f"Average ω :{(w1_avg_around)*180/np.pi}degree/s \n" ,
          f"Average t :{((T1.sum())/len(T1))}s \n",
          f"Average T :{(2*np.pi)/(w1_avg_around)}s \n",
          f"Average f :{(w1_avg_around)/(2*np.pi)}1/s \n",
          f"Total θ :{w1_avg_around*T1[-1]}rad \n",
          f"Area in the graph:{w1_avg*T1[-1]} rad")
    print(f"For the 2nd sub-set of {sheet} \n",
          f"Average S :{np.average(S2)}#lines/s \n",
          f"Average ω :{w2_avg_around}rad/s \n",
          f"Average ω :{(w2_avg_around)/(2*np.pi)}rev/s \n",
          f"Average ω :{(w2_avg_around)*180/np.pi}degree/s \n" ,
          f"Average t :{((T2.sum())/len(T2))}s \n",
          f"Average T :{(2*np.pi)/(w2_avg_around)}s \n",
          f"Average f :{(w2_avg_around)/(2*np.pi)}1/s \n",
          f"Total θ :{w2_avg_around*T2[-1]}rad \n",
          f"Area in the graph:{w2_avg*T2[-1]} rad")