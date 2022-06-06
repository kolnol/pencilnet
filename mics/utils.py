from cProfile import label
from turtle import color
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

df = pd.read_csv("PencilNet/Public_datasets/Trained_models/pencilnet-2022-01-22-13-45/history.csv")
print("Contents in csv file:\n", df)
plt.plot(df['epoch'], df['loss'], label='train loss', color='blue')
plt.plot(df['epoch'], df['val_loss'], label='validation loss', color='red')
plt.show()