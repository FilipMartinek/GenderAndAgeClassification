import cv2
import sys
import numpy as np
from tensorflow import keras
import os
import random
import process_data


#USE GUIDE
#python3.8 determine_face/test.py [optional: MODEL_NAME] [optional (in order):  RES] [optional (in order): COLOR_TYPE] [optional (in order): DATASET]


#get arguments
temp = 1
try:
    MODEL_NAME = sys.argv[temp]
    temp += 1
except IndexError:
    MODEL_NAME = "model0"
try:
    RES = int(sys.argv[temp])
    temp += 1
except IndexError:
    RES = 32
try:
    COLOR_TYPE = sys.argv[temp]
    temp += 1
except IndexError:
    COLOR_TYPE = "bgr"
try:
    DATASET = sys.argv[temp]
except:
    DATASET = "UTKFace" #UTKFace or IMDB_WIKI


#define filedir and load model
filedir = os.getcwd()
Model = keras.models.load_model(f"determine_face/models/{MODEL_NAME}.h5")
print(Model.layers)


#test method
def test(index, images, images_original, Model):

    #load image
    image_test = images[index]

    #get predictions
    pred = Model.predict(np.array([image_test]))
    print(pred)
    gender_options = ["Male","Female"]
    age = int(np.round(pred[1][0] * 100))
    gender = int(np.round(pred[0][0]))

    #print predictions and save them for the window name
    print("Predicted Age: " + str(age))
    print("Predicted Gender: "+ gender_options[gender])
    win_name = f"prediction: {gender_options[gender]} {str(age)} years old"

    #show image and wait 5 secs
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    try:
        cv2.imshow(win_name, images_original[index]) #try to show origial image
    except Exception: #change later
        cv2.imshow(win_name, images[index])
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    #try to load original images
    try:
        images_original = np.load(f"{filedir}datasets/processed_data/images_original_{RES}_{COLOR_TYPE}_{DATASET}.npy")
    except FileNotFoundError:
        images_original = 0
    

    #load dataset
    if DATASET == "UTKFace":
        images2, images, labels2, labels = process_data.process_data_UTKFace(RES, COLOR_TYPE)
    elif DATASET == "IMDB_WIKI":
        images2, images, labels2, labels = process_data.process_data_IMDB_WIKI(RES, COLOR_TYPE)


    #test data
    for i in range(10):
        test(random.randrange(0, len(images) - 1), images, images_original, Model)