import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae


tar_og = pd.read_csv('targets_2s_final.csv')
tar_og = tar_og.drop(['Unnamed: 0'], axis=1)
print(tar_og)

p05_og = pd.read_csv('p05_2s_final.csv')
p05_og = p05_og.drop(['Unnamed: 0'], axis=1)

p50_og = pd.read_csv('p50_2s_final.csv')
p50_og = p50_og.drop(['Unnamed: 0'], axis=1)

p95_og = pd.read_csv('p95_2s_final.csv')
p95_og = p95_og.drop(['Unnamed: 0'], axis=1)

# list of integers of actual and calculated


def calculate_mae(timestep):
    # calculate MAE
    error = mae(tar_og[timestep], p50_og[timestep])

    # display
    print(timestep +" | Mean absolute error : " + str(error))

for i in range(20):
    calculate_mae("t+" +str(i))

# calculate_mae("t+3")


# frequency histogram graph
def histogram(timestep):
    bins_1 = np.arange(1, 15, 0.1).tolist()

    tar[timestep].plot(kind='hist', bins=bins_1, label = "target", title="Timestep: " + timestep)
    # p05[timestep].plot(kind='hist', bins=bins_1, label = "p05")
    p50[timestep].plot(kind='hist', bins=bins_1, label = "p50")
    # p95[timestep].plot(kind='hist', bins=bins_1, label = "p95")

    plt.xlabel("Headway")
    plt.legend(loc="upper right")
    # plt.show()

    plt.savefig('D:/Users/Mark Navalta/r/here/'+ timestep + '.png')
    plt.close()

# for i in range(0, 20, 1):
#     histogram("t+" +str(i))

# histogram("t+49")

# random vehicle from 2325 to 3366
import random
def select_car(tar, p05, p50, p95):
    found = 0
    while found != 1:
        ran_car = random.randint(2325, 3366)
        print("Car ID: ", ran_car)

        if ran_car in set(tar['identifier']):
            tar = tar.loc[tar["identifier"]==ran_car]
            # tar = tar.drop(['identifier'], axis=1)

            p05 = p05.loc[p05["identifier"]==ran_car]
            # p05 = p05.drop(['identifier'], axis=1)

            p50 = p50.loc[p50["identifier"]==ran_car]
            # p50 = p50.drop(['identifier'], axis=1)

            p95 = p95.loc[p95["identifier"]==ran_car]
            # p95 = p95.drop(['identifier'], axis=1)
            return tar, p05, p50, p95, ran_car
        else:
            print(ran_car, " does not exist")


def line_graph(tar, p05, p50, p95, interval, timestep, ran_car):
    # line graph
    tar = tar.iloc[::interval, :]
    p05 = p05.iloc[::interval, :]
    p50 = p50.iloc[::interval, :]
    p95 = p95.iloc[::interval, :]

    ax = tar.plot(x='forecast_time', y=timestep, kind='line', marker="o", markersize=3, color="black", label="target")
    p05.plot(ax=ax, x='forecast_time', y=timestep, kind='line', marker="o", markersize=3, color="red", label="p05", alpha=0.5)
    p50.plot(ax=ax, x='forecast_time', y=timestep, kind='line', marker="o", markersize=3, color="green", label="p50", alpha=0.5)
    p95.plot(ax=ax, x='forecast_time', y=timestep, kind='line', marker="o", markersize=3, color="blue", label="p95", alpha=0.5)
    plt.fill_between(p05['forecast_time'], p05[timestep], p50[timestep], alpha=0.2, color="brown")
    plt.fill_between(p50['forecast_time'], p50[timestep], p95[timestep], alpha=0.2, color="green")
    plt.title(str(ran_car) +" | " + timestep +" | 2s intervals")
    plt.ylabel("Time Headway")
    plt.xlabel("time")
    plt.xticks(rotation=15)
    plt.xticks(fontsize=7)
    # plt.xlim(0, 30)
    # plt.ylim(2, 4)
    ax = plt.gca()
    # recompute the ax.dataLim
    ax.relim()
    # update ax.viewLim using the new dataLim
    ax.autoscale_view()
    # plt.show()
    plt.savefig('D:/Users/Mark Navalta/r/here/'+ str(ran_car) + '_'+timestep+'.png')
    plt.close()


# line_graph(tar, p05, p50, p95, 20, "t+5")
# for i in range(0, 20, 1):
#     tar, p05, p50, p95, ran_car = select_car(tar, p05, p50, p95)
#     line_graph(tar, p05, p50, p95, 20, "t+" +str(i),ran_car)


def pred(tar, p05, p50, p95, ran_car):
    instance = 20
    bin = np.arange(-20, instance, 1).tolist()
    x = np.arange(0,instance,1).tolist()
    pre_x = np.arange(-20,0,1).tolist()
    pre_tar = tar.iloc[instance-20]
    tar = tar.iloc[instance]
    p05 = p05.iloc[instance]
    p50 = p50.iloc[instance]
    p95 = p95.iloc[instance]

    plt.title("20 step prediction")
    plt.xticks(bin)
    plt.plot(pre_x, pre_tar.values[2:], marker="o", markersize=2, color="black")

    plt.plot(x, tar.values[2:], marker="o", markersize=3, color="black", label="target")
    plt.plot(x, p05.values[2:], marker="o", markersize=3, color="red", label="p05")
    plt.plot(x, p50.values[2:], marker="o", markersize=3, color="green", label="p50")
    plt.plot(x, p95.values[2:], marker="o", markersize=3, color="blue", label="p95")
    plt.fill_between(x, (p05.values[2:]).astype(float), (p50.values[2:]).astype(float), alpha=0.2, color="brown")
    plt.fill_between(x, (p50.values[2:]).astype(float), (p95.values[2:]).astype(float), alpha=0.2, color="green")
    plt.title("20 step prediction | car:" + str(ran_car))
    plt.ylabel("Time Headway")
    plt.xlabel("Time Steps")
    plt.xticks(rotation=90)
    plt.xticks(fontsize=8)
    plt.legend(loc="upper left")

    # plt.show()
    plt.savefig('D:/Users/Mark Navalta/r/here/'+ str(ran_car) + '_20step.png')
    plt.close()

# prediction(tar, p05, p50, p95, 10, 2325, 21)

# for i in range(0, 20, 1):
#     tar, p05, p50, p95, ran_car = select_car(tar_og, p05_og, p50_og, p95_og)
#     pred(tar, p05, p50, p95, ran_car)



