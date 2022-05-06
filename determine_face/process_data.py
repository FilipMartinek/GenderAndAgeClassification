import sys, os, cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

#USE GUIDE:
#python3.8 determine_face/process_data.py [optional: "ow" for overwrite] [optional: DATASET] [optional (in order): MAXLEN] [optional (in order): RES] [optional (in order): COLORTYPE]
#DATASET can be UTKFace or IMDB_WIKI
#COLORTYPE can be bgr, rgb or bw

#initialize dir variables
filedir = os.getcwd()


#process and return data using the UTKFace dataset
def process_data_UTKFace(RES=32, COLORTYPE="bgr", OW=False, MAXLEN=30000):
    
    #output variables
    images = []
    # images_original = []  #optional, takes up a lot of space
    labels = []
    
    
    #try to load saved data
    try:
        if not(OW):
            images = np.load(f"{filedir}/datasets/processed_data/images_{RES}_{COLORTYPE}_UTKFace.npy")
            # images_original = np.load(f"{basedir}/datasets/processed_data/images_original_{RES}_{COLORTYPE}_UTKFace.npy")
            labels = np.load(f"{filedir}/datasets/processed_data/labels_{RES}_{COLORTYPE}_UTKFace.npy")
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        #get all files in the UTKFace folder
        files = os.listdir(f"{filedir}/datasets/UTKFace")

        #loop through all files
        for i, file in enumerate(files):
            #end if i is over the limit of pictures
            if i >= MAXLEN:
                break

            print(file)

            #save the image and get the age + gender from the filename
            age = int(file.split("_")[0])
            gender = int(file.split("_")[1])
            image = cv2.imread(f"{filedir}/datasets/UTKFace/{file}")
            
            #save original image
            # images_original.append(image)

            #process and save image
            if COLORTYPE == "bgr":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif COLORTYPE == "bw":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.resize(image, (RES, RES))
            images.append(image)

            #save label
            labels.append([[age/100.0], [gender]]) #divide age by hundred, so that the neural net gets a value between 0 and 1


        #convert data to numpy arrays and save them
        images = np.array(images)
        labels = np.array(labels)
        # images_original = np.array(images_original)
        np.save(f"{filedir}/datasets/processed_data/images_{RES}_{COLORTYPE}_UTKFace.npy", images)
        np.save(f"{filedir}/datasets/processed_data/labels_{RES}_{COLORTYPE}_UTKFace.npy", labels)
        # np.save(f"{basedir}/datasets/processed_data/images_original_{RES}_{COLORTYPE}_UTKFace.npy", images_original)
    
    #prepare data for neural network
    images_formatted = images / 255.0 #divide image data by 255, so that the neural nets gets all data between 0 and 1
    imgs_train, imgs_test, labels_train, labels_test = train_test_split(images_formatted, labels, test_size=0.3) #split data between testing and training data
    labels_train = [labels_train[:, 1], labels_train[:, 0]] #split data into 2 numpy arrays, one for all the ages and one ---\
    labels_test = [labels_test[:, 1], labels_test[:, 0]]    #for all the genders, so that the neural net can work with it  <-/

    #return data
    return (imgs_train, imgs_test, labels_train, labels_test)


#process and return data using the IMDB_wiki dataset
def process_data_IMDB_WIKI(RES=32, COLORTYPE="bgr", OW=False, MAXLEN=30000):
        
    #output variables
    images = []
    # images_original = []  #optional, takes up a lot of space
    labels = []    
    
    
    #try to load saved data
    try:
        if not(OW):
            images = np.load(f"{filedir}/datasets/processed_data/images_{RES}_{COLORTYPE}_IMDB_WIKI.npy")
            # images_original = np.load(f"{basedir}/datasets/processed_data/images_original_{RES}_{COLORTYPE}_UTKFace.npy")
            labels = np.load(f"{filedir}/datasets/processed_data/labels_{RES}_{COLORTYPE}_IMDB_WIKI.npy")
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        
        #load data info from .mat files
        imdb = loadmat(f"{filedir}/datasets/imdb_crop/imdb.mat")["imdb"]
        imdb_paths = imdb[0][0][2][0]
        imdb_genders = imdb[0][0][3][0]
        wiki = loadmat(f"{filedir}/datasets/wiki_crop/wiki.mat")["wiki"]
        wiki_paths = wiki[0][0][2][0]
        wiki_genders = wiki[0][0][3][0]


        #loop through all files in imdb
        for i in range(len(imdb_paths)):
            #end if i is over the limit of pictures
            if i >= MAXLEN:
                break

            #try, if there is an error, print it and skip the image
            try:
                #go from the end, so that there are enough females in the data
                if i >= MAXLEN/2:
                    i = len(imdb_paths) - i

                
                #get image and it's path
                image_filename = f"{filedir}/datasets/imdb_crop/{imdb_paths[i][0]}"
                print(image_filename)
                image = cv2.imread(image_filename)

                #save original image
                # images_original.append(image)

                #process and save image
                if COLORTYPE == "bgr":
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                elif COLORTYPE == "bw":
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                image = cv2.resize(image, (RES, RES))


                #get gender and swap 0 and 1, so that it is 0 for male and 1 for female
                gender = imdb_genders[i] * -1 + 1

                #get age from photo date and birthdate in filename
                birthdate = int(image_filename.split("_")[3].split("-")[0])
                pic_date = int(image_filename.split("_")[4].split(".")[0])
                age = pic_date - birthdate
                
                #save label and image
                labels.append([[age/100.0], [gender]]) #divide age by hundred, so that the neural net gets a value between 0 and 1
                images.append(image)

            except Exception as ex:
                print(f"Encounter an Exception: {ex}\nSkipping image")

        #loop through all files in wiki
        for i in range(len(wiki_paths)):
            #end if i + length of imdb is over the limit of pictures
            if i + len(imdb_paths) >= MAXLEN:
                break
            #try, if there is an error, print it and skip the image
            try:
                #get image and it's path
                image_filename = f"{filedir}/datasets/wiki_crop/{wiki_paths[i][0]}"
                print(image_filename)
                image = cv2.imread(image_filename)

                #save original image
                # images_original.append(image)

                #process image
                if COLORTYPE == "bgr":
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                elif COLORTYPE == "bw":
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                image = cv2.resize(image, (RES, RES))


                #get gender and swap 0 and 1, so that it is 0 for male and 1 for female
                gender = wiki_genders[i] * -1 + 1

                #get age from photo date and birthdate in filename
                birthdate = int(image_filename.split("_")[2].split("-")[0])
                pic_date = int(image_filename.split("_")[3].split(".")[0])
                age = pic_date - birthdate
                
                #save label and image
                labels.append([[age/100.0], [gender]]) #divide age by hundred, so that the neural net gets a value between 0 and 1
                images.append(image)

            except Exception as ex:
                print(f"Encounter an Exception: {ex}\nSkipping image")

        
        
        #convert data to numpy arrays and save them
        images = np.array(images)
        labels = np.array(labels)
        # images_original = np.array(images_original)
        np.save(f"{filedir}/datasets/processed_data/images_{RES}_{COLORTYPE}_IMDB_WIKI.npy", images)
        np.save(f"{filedir}/datasets/processed_data/labels_{RES}_{COLORTYPE}_IMDB_WIKI.npy", labels)
        # np.save(f"{basedir}/datasets/processed_data/images_original_{RES}_{COLORTYPE}_UTKFace.npy", images_original)
    
    #prepare data for neural network
    images_formatted = images / 255.0 #divide image data by 255, so that the neural nets gets all data between 0 and 1
    imgs_train, imgs_test, labels_train, labels_test = train_test_split(images_formatted, labels, test_size=0.3) #split data between testing and training data
    labels_train = [labels_train[:, 1], labels_train[:, 0]] #split data into 2 numpy arrays, one for all the ages and one ---\
    labels_test = [labels_test[:, 1], labels_test[:, 0]]    #for all the genders, so that the neural net can work with it  <-/

    #return data
    return (imgs_train, imgs_test, labels_train, labels_test)


#if program is run by itself
if __name__ == "__main__":

    #default parameters
    DATASET = "IMDB_WIKI"
    RES = 32
    COLORTYPE = "bgr"
    OW = False
    MAXLEN = sys.maxsize

    #get all the passed paramaters
    try:
        temp = 1
        if sys.argv[temp] == "ow":
            OW = True
            temp += 1
        if sys.argv[temp] == "UTKFace" or sys.argv[temp] == "IMDB_WIKI":
            DATASET = sys.argv[temp]
            temp += 1
        try:
            MAXLEN = int(sys.argv[temp])
            temp += 1
        except ValueError:
            pass
        
        RES = int(sys.argv[temp])
        temp += 1
        COLORTYPE = sys.argv[temp]
    except IndexError:
        pass
    
    #process data
    if DATASET == "UTKFace":
        process_data_UTKFace(RES, COLORTYPE, OW, MAXLEN)
    elif DATASET == "IMDB_WIKI":
        process_data_IMDB_WIKI(RES, COLORTYPE, OW, MAXLEN)