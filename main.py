import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from scipy.ndimage import uniform_filter,median_filter
from skimage.util import random_noise
import skimage.io as io
from os.path import exists
from keras.models import load_model
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn import decomposition
from skimage.util import random_noise
from skimage import img_as_float
from time import time
import scipy.fftpack as fp
import pywt
from tensorflow import keras
opt = keras.optimizers.Adam(learning_rate = 0.1)
from sklearn.decomposition import PCA

def preprocess(array,size = 28):
  size = array.shape[1]
  array= array.astype("float32")/255.0
  array=np.reshape(array,(len(array),size,size,1))
  return array

def noise(array):
  noise_factor=0.0005
  noise_array = array + noise_factor * np.random.normal(loc=0.0 , scale=1.0 , size=array.shape)
  return np.clip(noise_array,0.0,1.0)

def display(array1,array2,array3,array4,array5,sizes = 0):
  if(sizes == 0):
    sizes = array1.shape[1]
  n=10
  indices = np.random.randint(len(array1),size=n)
  images1 = array1[indices,:]
  images2 = array2[indices,:]
  images3 = array3[indices,:]
  images4 = array4[indices,:]
  images5 = array5[indices,:]

  plt.figure(figsize=(20,4))
  for i, (image1,image2,image3,image4,image5) in enumerate(zip(images1,images2,images3,images4,images5)):
    ##
    ax =plt.subplot(5,n,i+1) ##  5 radif , 10 khone , 0 - 10
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(image1.reshape(sizes,sizes))
    ##
    ax =plt.subplot(5,n,i+1+n) ##  5 radif , 10 khone , 10 - 20
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(image2.reshape(sizes,sizes))
    ##
    ax =plt.subplot(5,n,i+1+2*n) ##  5 radif , 10 khone , 10 - 20
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(image3.reshape(sizes,sizes))
    ##
    ax =plt.subplot(5,n,i+1+3*n)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(image4.reshape(sizes,sizes))
    ##
    ax =plt.subplot(5,n,i+1+4*n)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(image5.reshape(sizes,sizes))
    ##
  plt.show()

def load_data(mnist_or_faces):
  if mnist_or_faces == 'mnist':
    (train_data,_),(test_data,_) = mnist.load_data(path='./models/mnist.npz')
  else:
    dataset = fetch_olivetti_faces(shuffle=True, random_state=0)
    d1 = dataset['images'].reshape(*dataset['images'].shape,1)
    train_data = d1[:int(d1.shape[0]*0.2),...]
    test_data = d1[int(d1.shape[0]*0.2):,...]
  train_data = preprocess(train_data,mnist_or_faces)
  test_data = preprocess(test_data,mnist_or_faces)
  noisy_train_data = noise(train_data)
  noisy_test_data = noise(test_data)
  return train_data,test_data,noisy_train_data,noisy_test_data

def loadmodel(data):
  autoencoder = load_model(f'./models/{data}.h5')
  return autoencoder

def train(data,train_data,test_data,noisy_train_data,noisy_test_data,size = 0):
  size = train_data.shape[1]
  input = layers.Input(shape=(size,size,1))
  x = layers.Conv2D(32,(3,3),activation="relu",padding="same")(input)
  x = layers.MaxPooling2D((2,2),padding="same")(x)
  x = layers.Conv2D(32,(3,3),activation="relu",padding="same")(x)
  x = layers.MaxPooling2D((2,2),padding="same")(x)
  x = layers.Conv2DTranspose(32,(3,3),strides=2,activation="relu",padding="same")(x)
  x = layers.Conv2DTranspose(32,(3,3),strides=2,activation="relu",padding="same")(x)
  x = layers.Conv2D(1,(3,3),activation="sigmoid",padding="same")(x)
  autoencoder = Model(input,x)
  autoencoder.compile(optimizer=opt,loss="binary_crossentropy")
  autoencoder.summary()
  autoencoder.fit(
    x=noisy_train_data,
    y=train_data,
    epochs=20,
    batch_size=128,
    shuffle=True,
    validation_data=(test_data,test_data),
  )
  autoencoder.save(f'./models/{data}.h5')
  return autoencoder

def uniform(noisy_test_data):
  uniform_test_data = uniform_filter(noisy_test_data,2)
  return uniform_test_data


def pca(noisy_test_data,size = 0):
  n_components = 100 # 256
  estimator = PCA(n_components=n_components)
  flat_noisy_test_data = noisy_test_data.reshape(noisy_test_data.shape[0],-1)
  digitis_recon = estimator.inverse_transform(estimator.fit_transform(flat_noisy_test_data))
  pca_test_data = digitis_recon.reshape(noisy_test_data.shape[0],size,size,1)
  return pca_test_data


def learnit(data):
  if(exists('./models/{data}.h5')):
    autoencoder = train(data,train_data,test_data,noisy_train_data,noisy_test_data)
  else:
    autoencoder = loadmodel(data)
  return autoencoder
###### 


data='mnist'
train_data,test_data,noisy_train_data,noisy_test_data = load_data(data)
autoencoder = learnit(data)
predictions = autoencoder.predict(noisy_test_data)
uniform_test_data = uniform(noisy_test_data)
pca_test_data = pca(noisy_test_data,train_data.shape[2])
display(test_data,noisy_test_data,uniform_test_data,pca_test_data,predictions)