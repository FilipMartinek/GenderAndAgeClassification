import pickle, sys, os
from time import time
import matplotlib.pyplot as plt
from ann_visualizer.visualize import ann_viz
from tensorflow import keras

#USE GUIDE:
#python3.8 determine_face/show_train_data.py [key] [models separated by space]
#key can be e. g. val_loss

#initialize dir variables
filedir = os.getcwd()


#get paramaters histories
model_names = []
temp = 1

try:
    key = sys.argv[temp]
    temp += 1
except IndexError:
    key = "val_loss"

while True:
    try:
        model_names.append(sys.argv[temp])
        temp += 1
    except:
        break

#if model_names are empty
if len(model_names) == 0:
    model_names = ["model0", "model1", "model2", "model3", "model4"]


#get histories
histories = [pickle.load(open(f"{filedir}/determine_face/models/{model_name}_history.pickle", "rb")) for model_name in model_names]


#add all the elements to the chart
for i, history in enumerate(histories):
    plt.plot(history[key])

    #calculate and output time per epoch
    time_per_epoch = history["time"] / len(history["val_loss"])
    print(f"{model_names[i]} training time per epoch: {time_per_epoch}")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(model_names)
plt.title(key)
plt.show()