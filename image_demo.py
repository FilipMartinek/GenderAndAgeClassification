import os
import sys
import cv2
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def get_larger(a, b):
    if a >= b:
        return a
    return b

def visualize_conv(Model, input, layer_name, rows, cols):
    #get output layer
    conv_out = Model.get_layer(layer_name).output

    #create intermediate model and prediction
    inter_model = keras.models.Model(inputs=Model.input, outputs=conv_out)
    inter_pred = inter_model.predict(input)

    i = 0

    fig, ax = plt.subplots(rows, cols)

    for r in range(rows):
        for c in range(cols):
            ax[r][c].imshow(inter_pred[0, :, :, i])
            i += 1
    plt.show()


filedir = os.getcwd()

temp = 1
try:
    img_dir = sys.argv[temp]
    temp += 1
except IndexError:
    print("Please give the image filename argument (e.g.: python image_face_detection.py image_path.jpg)")

try:
    if sys.argv[temp] == "v":
        viz_convs = True
        temp += 1
    else:
        raise IndexError        
except IndexError:
    viz_convs = False

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
    temp += 1
except IndexError:
    COLOR_TYPE = "bgr"

try:
    PADDING = sys.argv[temp]
except IndexError:
    PADDING = 0

try:
    image = cv2.imread(filedir + img_dir)
except:
    image = cv2.imread(filedir + img_dir)



Model = keras.models.load_model(f"{filedir}/determine_face/models/{model_name}.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

gender_options = ["Male","Female"]

font = cv2.FONT_HERSHEY_PLAIN


gray_scale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("test", gray_scale_img)
# cv2.waitKey(5000)
# cv2.destroyAllWindows()

faces = face_cascade.detectMultiScale(gray_scale_img, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)
print(faces)

for (x, y, width, height) in faces:

    #draw rect around face
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 1)
    a = get_larger(height, width)


    #save and process cropped image
    processed_img = image[y:(y + a), x:(x + a)]
    cv2.imshow("win", processed_img)
    # processed_img = image[y:(y + height), x:(x + width)]
    if COLOR_TYPE == "bgr":
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
    elif COLOR_TYPE == "bw":
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
    processed_img = cv2.resize(processed_img, (RES, RES))
    processed_img = np.array([processed_img])/255.0


    #get predictions
    pred = Model.predict(processed_img)

    if viz_convs:
        visualize_conv(Model, processed_img, "conv2d", 4, 8)
        visualize_conv(Model, processed_img, "conv2d_1", 4, 16)
        visualize_conv(Model, processed_img, "conv2d_2", 8, 16)
        visualize_conv(Model, processed_img, "conv2d_3", 8, 32)
    age = int(np.round(pred[1][0] * 100))
    gender = int(np.round(pred[0][0]))

    #print predictions above rect
    text = f"Gender: {gender_options[gender]} | Age: {age}"
    cv2.putText(image, text, (x, y - 3), font, 0.75, (0, 255, 0), 1, cv2.LINE_AA)

#show image
cv2.imshow("Face recognition - press ESC to quit", image)

while True:
    #exit
    key = cv2.waitKey(1)
    if key % 256 == 27: #ESC key
        cv2.destroyAllWindows()
        break
