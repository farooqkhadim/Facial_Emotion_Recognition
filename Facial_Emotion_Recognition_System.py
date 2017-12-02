
# coding: utf-8

# # Facial Emotion Recognition System
# ### Using TFLearn Api for this purpose rather than making each convolutional layer and edit them
# ### Use tensorboard to show statistics
# 

# In[1]:

#important dependencies to import
import numpy as np # for multi-dimensional arrays
import os # for accessing directories for local data
from random import shuffle # shuffling data for training and testing
from tqdm import tqdm # It will show data to iterate in arrays or multi-arrays on console
import cv2 # used to read images
import pandas as pd # used to read csv file and create dataframe




# In[ ]:

train_directory = 'https://www.floydhub.com/farooqkhadim/datasets/kaggle-fer-dataset/3/FER2013Train'
train_directory = 'https://www.floydhub.com/farooqkhadim/datasets/kaggle-fer-dataset/3/FER2013Test'

img_size = 48 # as the images size are the same as 48 * 48 but we are fixing it for our check

learning_rate = 1e-3 #learning rate. If the learning rate is slow. It gets good results.
# Give name to model. As it take time to train model and require hardware so to avoid this expense,
# we are saving in order to retreive it later for further use.
model_name = 'facialrecognition-{}-{}.model'.format(LR, '2conv-basic')


# In[ ]:

# reading labeltest.csv file which contains emotions for test using pandas read_csv function
df_test = pd.read_csv('https://www.floydhub.com/farooqkhadim/datasets/kaggle-fer-dataset/3/LabelTest.csv', header=None)
# validating or checking by checking statistics
df_test.describe()
# show some of the values
df_test.head()
# converting dataframe to numpy nd array using pandas.dataframe function as_matrix()
test_label = df_test.as_matrix()


# In[ ]:

def create_train_data():
    #Function to create training data
    # Again we are reading csv file through pandas read_csv() 
    df_train = pd.read_csv('https://www.floydhub.com/farooqkhadim/datasets/kaggle-fer-dataset/3/LabelTrain.csv', header=None)
    # convert dataframe to ndarray
    train_label = df_train.as_matrix()
    training_data = []
    # iterating through the directories of training and through array to get values and add them to single array so
    # feature and label gets togather i.e [features: label]
    for img,label in zip(tqdm(os.listdir(train_directory)),np.nditer(train_label)):
        path = os.path.join(train_directory,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size,img_size))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data) # suffling data
    #saving created array for further use and we don't need to iterate through again to make traing data
    np.save('train_data.npy', training_data)
    return training_data


# In[ ]:

def process_test_data():
    #Function to create testing data
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        img_num = img.split('.')[0]
        path = os.path.join(TEST_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data



# In[ ]:

#callig create train data to
train_data = create_train_data()


# In[ ]:

# import tflearn high level api and setting default graph for tensorflow
import tflearn
tf.reset_default_graph()


# In[ ]:

# import different convolutional layers and functions
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


# In[ ]:

# creating layers of Convolutional Neural Network
# by increasing convolutional layers its accuracy increases to some point but its processing time also increases and
# you have to get extra gpu for this
convnet = input_data(shape=[None, img_size, img_size, 1], name='input')

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 156, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1800, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 6, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy', name='targets')


# In[ ]:

# creating model and it's log is stored in tensorboad directory
model = tflearn.DNN(convnet, tensorboard_dir='log')


# In[ ]:

# incase we already trained model and we can get from it rather than running whole process again
if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('Already processed model loaded!')


# In[ ]:

# getting certain limited values for testing and training
train = train_data[:-1800]
test = train_data[-1800:]


# In[ ]:

# X and Y values. X for features and Y for labels(emotion)
X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]


# In[ ]:

# getting test x and test y
test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]


# In[ ]:

# fitting model and running
model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=model_name)
model.save(model_name)

