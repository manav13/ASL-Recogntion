# Predicting Classes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
model = joblib.load('ASL-CNN')

test_set = pd.read_csv("sign_mnist_test.csv").values
testX = test_set[:,1:].reshape(test_set.shape[0],28, 28, 1).astype( 'float32' )
test_set_data = np.array(test_set)
label = test_set_data[:,0]
images = test_set_data[:,[range(1,785)]]

for i in range(len(testX)):
    x = np.reshape(testX[i],[1,28,28,1])
    classes = model.predict_classes(x)
    img = images[i].reshape(28,28)
    plt.imshow(img, cmap='gray')
    plt.show()
    print("Original Image Label: ",test_set[i][0],", Original Character: ",chr(test_set[i][0]+ord("a")))
    if(classes>=9):
        print ("Predicted Label: ",classes+1,", Predicted Character: ",chr(classes+1+ord("a")))
    else:
        print ("Predicted Label: ",classes,", Predicted Character: ",chr(classes+ord("a")))
    print("Press any key to continue and q to exit: ")
    j = input()
    if(j=="q"):
        break