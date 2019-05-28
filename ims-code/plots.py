"""
Generate plot
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()
from etl import *


#main_path = "../data/IMS/{}".format(experiment)

#all_files = get_all_files(main_path)
#cols = ["x1", "y1", "x2", "y2","x3", "y3", "x4", "y4"]
#file = all_files[-1]
#datetime = file.split("/")[-1]
#hour = ":".join(datetime.split(".")[3:])
#date = "-".join(datetime.split(".")[:3])
#new_date = "{} {}".format(date,hour)
#df = get_dataframe(file)
#df.columns = cols
#x1 = df[0].values; x2 = df[2].values; x3 = df[4].values; x4 = df[6].values
#y1 = df[1].values; y2 = df[3].values; y3 = df[5].values; y4 = df[7].values
#x = np.linspace(0,10,len(x1))
#y = y1
#title = "Bearing 1 axial vibration measured at {}".format(new_date)
#df_new = pd.DataFrame({"Time in minute":x, "Axial vibration":y})
#ax = sns.lineplot(x="Time in minute", y="Axial vibration",data=df_new)
#ax.legend( labels=["A","B"])
#plt.plot(x1,y1)
#plt.plot(y2)

#plt.title(title)
#plt.savefig("../articles/thesis-pictures/bearing1-test1.png")

#plt.show()

def save_bearing_plot(exp_numb, bearing_nb, save=True):
    experiments = {"1": "1st_test",
                   "2": "2nd_test",
                   "3": "3rd_test"}
    experiment = experiments["{}".format(exp_numb)]
    main_path = "../data/IMS/{}".format(experiment)
    all_files = get_all_files(main_path)
    file = all_files[-3]
    datetime = file.split("/")[-1]
    hour = ":".join(datetime.split(".")[3:])
    date = "-".join(datetime.split(".")[:3])
    new_date = "{} {}".format(date,hour)
    df = get_dataframe(file)
    lim = 1000
    x = np.linspace(0,1,len(df[bearing_nb].values))[:lim]
    y = df[bearing_nb].values[:lim]
    title = "Test {}. Bearing {} axial vibration measured at {}".format(exp_numb, bearing_nb+1, new_date)
    df_new = pd.DataFrame({"Time in second":x, "Axial vibration":y})
    ax = sns.lineplot(x="Time in second", y="Axial vibration",data=df_new)
    plt.title(title)
    if save:
        plt.savefig("../articles/thesis-pictures/bearing1-iqr{}.png".format(exp_numb))
    else:
        plt.show()



if __name__ == '__main__':

    exp_numb = 2
    bearing_nb = 0
    save_bearing_plot(exp_numb,bearing_nb,save=False)

















#
