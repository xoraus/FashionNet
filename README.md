# How to build to a Fashion Classifier in 5 easy steps using Deep Learning.



[Sajjad Salaria](https://medium.com/@xoraus?source=post_page-----c0817b615ef8----------------------)

[Feb 15](https://medium.com/datadriveninvestor/how-to-build-to-a-fashion-classifier-in-5-easy-steps-using-deep-learning-c0817b615ef8?source=post_page-----c0817b615ef8----------------------)  ·  6  min read

_Every weekend I build a machine learning project to get my hands dirty in data science, this week I picked MNIST Dataset._

> “You can have anything you want in life if you dress for it.” — Edith Head.

_so, buckle up for some code, memes and a little bit of theory._

[https://newsladder.net/fashion-a-never-ending-cycle/](https://newsladder.net/fashion-a-never-ending-cycle/)

# STEP 1: PROBLEM STATEMENT AND BUSINESS CASE

The fashion training set consists of 70,000 images divided into 60,000 training and 10,000 testing samples. Dataset sample consists of 28x28 grayscale image, associated with a label from 10 classes.

> The 10 classes are as follows:  
> 0 => T-shirt/top 1 => Trouser 2 => Pullover 3 => Dress 4 => Coat 5 => Sandal 6 => Shirt 7 => Sneaker 8 => Bag 9 => Ankle boot

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255.

![Image result for KERAS MEMES](https://miro.medium.com/proxy/1*QbekpmNE8lCvSQzHLTPDHQ.png)

[https://becominghuman.ai/param-ishan-for-deep-learning-101-d6f049970584?gi=686405f4538d](https://becominghuman.ai/param-ishan-for-deep-learning-101-d6f049970584?gi=686405f4538d)

But before diving into the Coding, let’s first understand what an artificial neural network is, An  **artificial neural network**  is an interconnected group of nodes, similar to the vast  **network**  of neurons in  **a**  brain. Here, each circular node represents an  **artificial neuron**  and an arrow represents  **a**  connection from the output of one  **artificial neuron**  to the input of another.

![](https://miro.medium.com/max/25/0*8gKiHcB0ZwNmqi6i.png?q=20)

![](https://miro.medium.com/max/300/0*8gKiHcB0ZwNmqi6i.png)

[https://en.wikipedia.org/wiki/File:Colored_neural_network.svg](https://en.wikipedia.org/wiki/File:Colored_neural_network.svg)

The neurons collect signals from input channels named dendrites, processes information in its nucleus and then generates an output in a long thin branch called axon. Human learning occurs adaptively by varying the bond strength between these neurons.

![](https://miro.medium.com/max/30/0*X24ORhTkVU7Zpe0S.png?q=20)

![](https://miro.medium.com/max/300/0*X24ORhTkVU7Zpe0S.png)

[https://simple.wikipedia.org/wiki/File:Neuron.svg](https://simple.wikipedia.org/wiki/File:Neuron.svg)

# STEP 2: IMPORTING DATA

# import libraries   
import pandas as pd # Import Pandas for data manipulation using dataframes  
import numpy as np # Import Numpy for data statistical analysis   
import matplotlib.pyplot as plt # Import matplotlib for data visualisation  
import seaborn as sns  
import random

## Data frames creation for both training and testing datasets

fashion_train_df = pd.read_csv(‘input/fashion-mnist_train.csv’,sep=’,’)  
fashion_test_df = pd.read_csv(‘input/fashion-mnist_test.csv’, sep = ‘,’)

# STEP 3: VISUALIZATION OF THE DATASET

Let’s view the head of the training dataset

# 784 indicates 28x28 pixels and 1 coloumn for the label  
# After you check the tail, 60,000 training dataset are present  
fashion_train_df.head()

![](https://miro.medium.com/max/30/1*o6g-cGPnuDry-F9DE_7dhg.png?q=20)

![](https://miro.medium.com/max/1008/1*o6g-cGPnuDry-F9DE_7dhg.png)

df.head()

# Let's view the last elements in the training dataset  
fashion_train_df.tail()

![](https://miro.medium.com/max/30/1*FKmEdPKnPsua6Rv6gGGHTA.png?q=20)

![](https://miro.medium.com/max/1017/1*FKmEdPKnPsua6Rv6gGGHTA.png)

df.tail()

similarly for testing data

# Let’s view the head of the testing dataset  
fashion_test_df.head()# Let's view the last elements in the testing dataset  
fashion_test_df.tail()fashion_train_df.shape  
(60000, 785)

## Create training and testing arrays

training = np.array(fashion_train_df, dtype = ‘float32’)  
testing = np.array(fashion_test_df, dtype=’float32')

## Let’s view some images!

i = random.randint(1,60000) # select any random index from 1 to 60,000  
plt.imshow( training[i,1:].reshape((28,28)) ) # reshape and plot the imageplt.imshow( training[i,1:].reshape((28,28)) , cmap = 'gray') # reshape and plot the image# Remember the 10 classes decoding is as follows:  
# 0 => T-shirt/top  
# 1 => Trouser  
# 2 => Pullover  
# 3 => Dress  
# 4 => Coat  
# 5 => Sandal  
# 6 => Shirt  
# 7 => Sneaker  
# 8 => Bag  
# 9 => Ankle boot

![](https://miro.medium.com/max/30/1*TFWLKpqYAdc7TnFQM4kVBA.png?q=20)

![](https://miro.medium.com/max/390/1*TFWLKpqYAdc7TnFQM4kVBA.png)

shirt

## Let’s view more images in a grid format

W_grid = 15  
L_grid = 15# fig, axes = plt.subplots(L_grid, W_grid)  
# subplot return the figure object and axes object  
# we can use the axes object to plot specific figures at various locationsfig, axes = plt.subplots(L_grid, W_grid, figsize = (17,17))axes = axes.ravel() # flaten the 15 x 15 matrix into 225 arrayn_training = len(training) # get the length of the training dataset# Select a random number from 0 to n_training  
for i in np.arange(0, W_grid * L_grid): # create evenly spaces variables# Select a random number  
    index = np.random.randint(0, n_training)  
    # read and display an image with the selected index      
    axes[i].imshow( training[index,1:].reshape((28,28)) )  
    axes[i].set_title(training[index,0], fontsize = 8)  
    axes[i].axis('off')plt.subplots_adjust(hspace=0.4)

![](https://miro.medium.com/max/30/1*UyXeI5_cf2aZhjdJmLhtlQ.png?q=20)

![](https://miro.medium.com/max/977/1*UyXeI5_cf2aZhjdJmLhtlQ.png)

images in a grid format

# STEP 4: TRAINING THE MODEL

## Prepare the training and testing dataset

X_train = training[:,1:]/255  
y_train = training[:,0]X_test = testing[:,1:]/255  
y_test = testing[:,0]from sklearn.model_selection import train_test_splitX_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size = 0.2, random_state = 12345)  

## unpack the tuple

X_train = X_train.reshape(X_train.shape[0], *(28, 28, 1))  
X_test = X_test.reshape(X_test.shape[0], *(28, 28, 1))  
X_validate = X_validate.reshape(X_validate.shape[0], *(28, 28, 1))

# What the Keras you’re talking about!

![](https://miro.medium.com/max/30/0*yLRvbwmM_RYU7vXC.jpg?q=20)

![](https://miro.medium.com/max/750/0*yLRvbwmM_RYU7vXC.jpg)

[https://www.reddit.com/r/ProgrammerHumor/comments/a8ru4p/machine_learning_be_like/](https://www.reddit.com/r/ProgrammerHumor/comments/a8ru4p/machine_learning_be_like/)

import keras # open source Neural network library madke our life much easier# y_train = keras.utils.to_categorical(y_train, 10)  
# y_test = keras.utils.to_categorical(y_test, 10)cnn_model = Sequential()# Try 32 fliters first then 64  
cnn_model.add(Conv2D(64,3, 3, input_shape = (28,28,1), activation='relu'))  
cnn_model.add(MaxPooling2D(pool_size = (2, 2)))cnn_model.add(Dropout(0.25))# cnn_model.add(Conv2D(32,3, 3, activation='relu'))  
# cnn_model.add(MaxPooling2D(pool_size = (2, 2)))cnn_model.add(Flatten())  
cnn_model.add(Dense(output_dim = 32, activation = 'relu'))  
cnn_model.add(Dense(output_dim = 10, activation = 'sigmoid'))cnn_model.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001),metrics =['accuracy'])epochs = 50history = cnn_model.fit(X_train,  
                        y_train,  
                        batch_size = 512,  
                        nb_epoch = epochs,  
                        verbose = 1,  
                        validation_data = (X_validate, y_validate))

# STEP 5: EVALUATING THE MODEL

**Model evaluation**  metrics are  **used**  to assess goodness of fit between  **model**  and data, to compare different  **models**, in the context of  **model**  selection, and to predict how predictions (associated with a specific  **model**  and data set) are expected to be accurate

![](https://miro.medium.com/max/30/0*zCSaizPSZPOfe0Rc?q=20)

![](https://miro.medium.com/max/400/0*zCSaizPSZPOfe0Rc)

[https://hacktilldawn.com/2016/09/25/inception-modules-explained-and-implemented/](https://hacktilldawn.com/2016/09/25/inception-modules-explained-and-implemented/)

evaluation = cnn_model.evaluate(X_test, y_test)  
print('Test Accuracy : {:.3f}'.format(evaluation[1]))

Get the predictions for the test data

predicted_classes = cnn_model.predict_classes(X_test)L = 5  
W = 5  
fig, axes = plt.subplots(L, W, figsize = (12,12))  
axes = axes.ravel() #for i in np.arange(0, L * W):    
    axes[i].imshow(X_test[i].reshape(28,28))  
    axes[i].set_title("Prediction Class = {:0.1f}\n True Class = {:0.1f}".format(predicted_classes[i], y_test[i]))  
    axes[i].axis('off')plt.subplots_adjust(wspace=0.5)from sklearn.metrics import confusion_matrix  
cm = confusion_matrix(y_test, predicted_classes)  
plt.figure(figsize = (14,10))  
sns.heatmap(cm, annot=True)  
# Sum the diagonal element to get the total true correct valuesfrom sklearn.metrics import classification_reportnum_classes = 10  
target_names = ["Class {}".format(i) for i in range(num_classes)]print(classification_report(y_test, predicted_classes, target_names = target_names))

# References

[1]  [https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)

[2]  [https://www.pyimagesearch.com/2019/02/11/fashion-mnist-with-keras-and-deep-learning/](https://www.pyimagesearch.com/2019/02/11/fashion-mnist-with-keras-and-deep-learning/)

[3]  [https://pravarmahajan.github.io/fashion/](https://pravarmahajan.github.io/fashion/)
