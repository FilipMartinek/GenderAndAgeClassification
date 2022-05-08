import os
import cv2
import sys
from tensorflow import keras
import numpy as np


UPDATE_FRAME = 5

def get_larger(a, b):
    if a >= b:
        return a
    return b


temp = 1

try:
    model_name = sys.argv[temp]
    temp += 1
except IndexError:
    model_name = "model0"

try:
    RES = int(sys.argv[temp])
    temp += 1
except IndexError:
    RES = 48

try:
    COLOR_TYPE = sys.argv[temp]
except IndexError:
    COLOR_TYPE = "bgr"

filedir = os.getcwd()
Model = keras.models.load_model(f"{filedir}/determine_face/models/{model_name}.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "/haarcascade_frontalface_default.xml")
video_input = cv2.VideoCapture(0)

gender_options = ["Male","Female"]

font = cv2.FONT_HERSHEY_SIMPLEX


if not video_input.isOpened():
    print("Unable to access the camera")
    exit()


fram_c = UPDATE_FRAME

print("Streaming started")
#main loop
while True:

    ret, image = video_input.read()
    if not(ret):
        print("Can't receive frame, quitting...")
        break
    

    #convert to grayscale for face cascade
    gray_scale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray_scale_img,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, width, height) in faces:
        
        #draw rect around face
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 1)
        a = get_larger(height, width)


        #save and process cropped image
        processed_img = image[y:(y + a), x:(x + a)]
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
        processed_img = cv2.resize(processed_img, (RES, RES))


        #get predictions if the loop is at an update frame
        if fram_c >= UPDATE_FRAME:
            pred = Model.predict(np.array([processed_img])/255.0)
            age = int(np.round(pred[1][0] * 100))
            gender = int(np.round(pred[0][0]))
            fram_c = 0
        else:
            fram_c += 1

        #print predictions above rect
        text = f"Gender: {gender_options[gender]} | Age: {age}"
        cv2.putText(image, text, (x, y - 3), font, 0.75, (0, 255, 0), 1, cv2.LINE_AA)

    #show frame
    cv2.imshow("Face recognition - press ESC to quit", image)

    #exit when esc pressed
    key = cv2.waitKey(1)
    if key % 256 == 27: #ESC key
        break

#stio recording and destroy windows
video_input.release()
cv2.destroyAllWindows()