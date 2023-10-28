# FashionNet - A Fashion Classifier using Deep Learning

[![KERAS MEMES](https://miro.medium.com/proxy/1*QbekpmNE8lCvSQzHLTPDHQ.png)](https://miro.medium.com/proxy/1*QbekpmNE8lCvSQzHLTPDHQ.png)

Welcome to the Fashion Classifier project, where we'll build a fashion classifier using deep learning in five easy steps.

## Table of Contents

- [Introduction](#introduction)
- [Step 1: Problem Statement and Business Case](#step-1-problem-statement-and-business-case)
- [Step 2: Importing Data](#step-2-importing-data)
- [Step 3: Visualization of the Dataset](#step-3-visualization-of-the-dataset)
- [Step 4: Training the Model](#step-4-training-the-model)
- [Step 5: Evaluating the Model](#step-5-evaluating-the-model)
- [Appendix](#appendix)


## Introduction

> "You can have anything you want in life if you dress for it." — Edith Head.

In this project, we'll build a fashion classifier using Convolutional Neural Networks (CNNs), a class of deep learning models designed for image processing. CNNs consist of convolutional layers, pooling layers, and fully connected layers. We'll cover image preprocessing, loss functions, optimizers, model evaluation, overfitting prevention, data augmentation, hyperparameter tuning, and transfer learning.

## Step 1: Problem Statement and Business Case

The fashion training set consists of 70,000 images divided into 60,000 training and 10,000 testing samples. Each image is a 28x28 grayscale image associated with one of 10 classes. These classes include T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot.


## Step 2: Importing Data

```python
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Data frames creation for both training and testing datasets
fashion_train_df = pd.read_csv('input/fashion-mnist_train.csv', sep=',')
fashion_test_df = pd.read_csv('input/fashion-mnist_test.csv', sep=',')
```

## STEP 3: VISUALIZATION OF THE DATASET

#### Let’s view the head of the training dataset

```python
fashion_train_df.head()
fashion_train_df.tail()
```

![](https://miro.medium.com/max/30/1*o6g-cGPnuDry-F9DE_7dhg.png?q=20)

![](https://miro.medium.com/max/1008/1*o6g-cGPnuDry-F9DE_7dhg.png)

Similarly, for the testing dataset.

```python
fashion_test_df.head()
fashion_test_df.tail()
```

#### Now. let's view some images from the dataset.

```python
i = random.randint(1, 60000)
plt.imshow(training[i, 1:].reshape((28, 28)), cmap='gray')
```

![](https://miro.medium.com/max/30/1*FKmEdPKnPsua6Rv6gGGHTA.png?q=20)

![](https://miro.medium.com/max/1017/1*FKmEdPKnPsua6Rv6gGGHTA.png)


#### Create training and testing arrays

```python
training = np.array(fashion_train_df, dtype = ‘float32’)  
testing = np.array(fashion_test_df, dtype=’float32')
```

#### Let’s view some images!

```python
i = random.randint(1,60000) # select any random index from 1 to 60,000  
plt.imshow( training[i,1:].reshape((28,28)) ) # reshape and plot the imageplt.imshow( training[i,1:].reshape((28,28)) , cmap = 'gray') # reshape and plot the image# Remember the 10 classes decoding is as follows:  
```

0 => T-shirt/top  
1 => Trouser  
2 => Pullover  
3 => Dress  
4 => Coat  
5 => Sandal  
6 => Shirt  
7 => Sneaker  
8 => Bag  
9 => Ankle boot

![](https://miro.medium.com/max/30/1*TFWLKpqYAdc7TnFQM4kVBA.png?q=20)

![](https://miro.medium.com/max/390/1*TFWLKpqYAdc7TnFQM4kVBA.png)

shirt

 #### Let’s view more images in a grid format

```python
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
```

![](https://miro.medium.com/max/30/1*UyXeI5_cf2aZhjdJmLhtlQ.png?q=20)

![](https://miro.medium.com/max/977/1*UyXeI5_cf2aZhjdJmLhtlQ.png)

images in a grid format

## STEP 4: TRAINING THE MODEL

#### To prepare the training and testing dataset, we'll perform some data preprocessing.

```python
X_train = training[:, 1:] / 255
y_train = training[:, 0]

X_test = testing[:, 1:] / 255
y_test = testing[:, 0]

# Split the training data into training and validation sets
from sklearn.model_selection import train_test_split

X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2, random_state=12345)
```

#### Now, let's reshape the data for modeling.

```python
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_validate = X_validate.reshape(X_validate.shape[0], 28, 28, 1)
```

#### We'll create a convolutional neural network (CNN) model for fashion classification.

![](https://miro.medium.com/max/30/0*yLRvbwmM_RYU7vXC.jpg?q=20)

![](https://miro.medium.com/max/750/0*yLRvbwmM_RYU7vXC.jpg)

```python
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense

cnn_model = Sequential()

cnn_model.add(Conv2D(64, 3, 3, input_shape=(28, 28, 1), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.25))

cnn_model.add(Flatten())
cnn_model.add(Dense(output_dim=32, activation='relu'))
cnn_model.add(Dense(output_dim=10, activation='sigmoid'))

cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

epochs = 50
history = cnn_model.fit(X_train, y_train, batch_size=512, nb_epoch=epochs, verbose=1, validation_data=(X_validate, y_validate))
```

[https://www.reddit.com/r/ProgrammerHumor/comments/a8ru4p/machine_learning_be_like/](https://www.reddit.com/r/ProgrammerHumor/comments/a8ru4p/machine_learning_be_like/)

# STEP 5: EVALUATING THE MODEL

**Model evaluation**  metrics are  **used**  to assess goodness of fit between  **model**  and data, to compare different  **models**, in the context of  **model**  selection, and to predict how predictions (associated with a specific  **model**  and data set) are expected to be accurate

![](https://miro.medium.com/max/30/0*zCSaizPSZPOfe0Rc?q=20)

![](https://miro.medium.com/max/400/0*zCSaizPSZPOfe0Rc)


#### Now, it's time to evaluate the model's performance.

```python
evaluation = cnn_model.evaluate(X_test, y_test)
print('Test Accuracy: {:.3f}'.format(evaluation[1]))
```

#### Let's also get the predictions for the test data and visualize them.

```python
predicted_classes = cnn_model.predict_classes(X_test)

# Visualize the predictions
L = 5
W = 5
fig, axes = plt.subplots(L, W, figsize=(12, 12))
axes = axes.ravel()

for i in np.arange(0, L * W):
    axes[i].imshow(X_test[i].reshape(28, 28))
    axes[i].set_title("Prediction Class = {:0.1f}\nTrue Class = {:0.1f}".format(predicted_classes[i], y_test[i]))
    axes[i].axis('off')
```

##### Display a confusion matrix

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=True)
```

##### Print classification report

```python
from sklearn.metrics import classification_report
num_classes = 10
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_test, predicted_classes, target_names=target_names))
```

## Appendix:

### Convolutional Neural Networks (CNNs)

CNNs are a class of deep learning models specifically designed for image processing. They consist of convolutional layers, pooling layers, and fully connected layers. 

- **Convolutional Layers**: These layers apply filters to extract features from input images. The filters slide over the input image, capturing spatial patterns.
- **Pooling Layers**: Pooling layers reduce the spatial dimensions of the feature maps, which helps reduce computational complexity.
- **Dropout**: Dropout layers are used to prevent overfitting.

 They randomly deactivate a portion of neurons during training.

#### Image Preprocessing

- **Normalization**: Normalizing pixel values to a range between 0 and 1 helps in training convergence.
- **Reshaping**: Images are reshaped to fit the input size of the neural network.

#### Loss Functions and Optimizers

- **Loss Function**: The choice of loss function depends on the task. In this project, we use sparse categorical cross-entropy, which is suitable for multi-class classification.
- **Optimizer**: The Adam optimizer is used with a learning rate of 0.001 for parameter updates.

#### Model Evaluation

- Evaluation metrics include accuracy, precision, recall, and F1-score.
- The confusion matrix visually represents the classification results.
- A classification report provides a comprehensive view of model performance.

#### Overfitting and Regularization

- Overfitting occurs when the model performs well on the training data but poorly on unseen data. Techniques like dropout layers help prevent overfitting.

#### Data Augmentation

- Data augmentation techniques increase the diversity of the training dataset by applying transformations like rotation, scaling, and flipping.

#### Hyperparameter Tuning

- Finding optimal hyperparameters, such as batch size and number of epochs, is crucial for model performance.

#### Transfer Learning

- Transfer learning involves using pre-trained models and fine-tuning them for specific tasks. It can save training time and resources.


## Appendix

### Additional Resources

- [Deep Learning with Keras](https://keras.io/): Official documentation for the Keras deep learning framework.
- [TensorFlow](https://www.tensorflow.org/): An open-source machine learning framework that can be used alongside Keras for more advanced deep learning projects.
- [Scikit-learn](https://scikit-learn.org/stable/): A powerful machine learning library for various tasks, including data preprocessing and model evaluation.
- [Matplotlib](https://matplotlib.org/): A popular Python library for creating visualizations and plots.
- [Seaborn](https://seaborn.pydata.org/): An easy-to-use Python data visualization library that works well with Matplotlib.
- [Reddit - Machine Learning Humor](https://www.reddit.com/r/ProgrammerHumor/comments/a8ru4p/machine_learning_be_like/): A lighthearted take on the complexities of machine learning.

### Contributors

- [Sajjad Salaria](https://github.com/xoraus) - Project Lead

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgments

Special thanks to the authors and contributors of the following resources:

- [Zalando Research - Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [PyImageSearch - Fashion-MNIST with Keras and Deep Learning](https://www.pyimagesearch.com/2019/02/11/fashion-mnist-with-keras-and-deep-learning/)

