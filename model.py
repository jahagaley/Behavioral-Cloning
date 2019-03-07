import csv
import cv2
import numpy as np
from keras.models import Sequential 
from keras.layers import *
from sklearn.model_selection import train_test_split


def getArrays(lines):
    
    images = []
    measurements = [] 
    check = False
    delta = 0.2
    correction = {"center": 0, "left": delta, "right": 0 - delta}
    
    for line in lines:
        
        if not check:
            check = True
            continue
        for i in range(3):    
            source_path = line[i]
            filename = source_path.split('/')[-1]                               
            curr_path = "/opt/carnd_p3/data/IMG/"+ filename

            img = cv2.imread(curr_path)
            images.append(img)
        
        
            measurement = float(line[3]) + float(correction[lines[0][i]])
            measurements.append(measurement)
        
    X_train = np.array(images)
    y_train = np.array(measurements)
    
    
    return X_train, y_train
    
    
def getTrainingData():
    
    lines = []
    
    with open("/opt/carnd_p3/data/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
           
    X_train, y_train = getArrays(lines)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.15, random_state=42)   
    
    return X_train, X_test, y_train, y_test
    

def useNVIDIAModel(X_train, y_train):
    
    model = Sequential()
    # normalization
    model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=X_train[0].shape))
    
    # cropping layer
    model.add(Cropping2D(cropping=((70,25),(0,0))))
              
    # adding convultion layers
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2), activation="relu"))
    
    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2), activation="relu"))
    
    model.add(Convolution2D(48, 3, 3, border_mode='valid', subsample=(2, 2), activation="relu"))
    
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(2, 2), activation="relu"))
    
    # flattening
    model.add(Flatten())
    
    # fully connected layers
    model.add(Dense(100, activation="relu"))
    model.add(ELU())
    model.add(Dense(50, activation="relu"))
    model.add(ELU())
    model.add(Dense(10, activation="relu"))
    model.add(ELU())
    model.add(Dense(1))
    
    model.compile(loss="mse", optimizer="adam")
    
    model.fit(X_train, y_train, validation_split=0.2,shuffle=True, nb_epoch=20)
    
    model.save("model.h5")
     
    return model
    
def main():
    
    X_train, X_test, y_train, y_test = getTrainingData()
    
    model = useNVIDIAModel(X_train, y_train)
    
    print()
    score = model.evaluate(X_test, y_test, verbose=2)
    
    print("Testing loss for saved model: " + str(score))
    
    
if __name__ == "__main__":
    
    main()
    