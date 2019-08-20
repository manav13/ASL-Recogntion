# Model for Sign Language Recognition

# Importing all the libraries
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))

# Step 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Adding extra convolution layers
classifier.add(Conv2D(128, kernel_size=3, activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Conv2D(256, kernel_size=2, activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Step 3 - Flatten
classifier.add(Flatten())

# Step 4 - Fully Connected Layer
classifier.add(Dense(256, activation='relu'))
classifier.add(Dropout(0.3))
classifier.add(Dense(24, activation='softmax'))

# Compile the Model
classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Importing the Dataset
import pandas as pd
train = pd.read_csv('sign_mnist_train.csv').values
test = pd.read_csv('sign_mnist_test.csv').values

# Reshape and normalize training data
trainX = train[:, 1:].reshape(train.shape[0],28, 28, 1).astype( 'float32' )
X_train = trainX / 255.0

y_train = train[:,0]


# Reshape and normalize test data
testX = test[:,1:].reshape(test.shape[0],28, 28, 1).astype( 'float32' )
X_test = testX / 255.0

y_test = test[:,0]

# Encoding the output
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

# Training the dataset
classifier.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25)


# Predicting Classes
import numpy as np
import matplotlib.pyplot as plt

test_set = pd.read_csv("sign_mnist_test.csv")
test_set_data = np.array(test_set)
label = test_set_data[:,0]
images = test_set_data[:,[range(1,785)]]

for i in range(len(testX)):
    x = np.reshape(testX[i],[1,28,28,1])
    classes = classifier.predict_classes(x)
    img = images[i].reshape(28,28)
    plt.imshow(img, cmap='gray')
    plt.show()
    print("Original Image Label: ",test[i][0],", Original Character: ",chr(test[i][0]+ord("a")))
    if(classes>=9):
        print ("Predicted Label: ",classes+1,", Original Character: ",chr(classes+1+ord("a")))
    else:
        print ("Predicted Label: ",classes,", Original Character: ",chr(classes+ord("a")))
    print("Press any key to continue and q to exit: ")
    j = input()
    if(j=="q"):
        break