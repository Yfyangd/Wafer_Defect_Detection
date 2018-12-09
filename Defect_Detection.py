
# coding: utf-8

# In[ ]:


import os, cv2, keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

# manipulate with numpy,load with panda
import numpy as np
import pandas as pd

# data visualization
import cv2
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Data Import
def read_dataset(path):
    data_list = []
    label_list = []
    for dirPath, dirNames, fileNames in os.walk(os.path.normpath(path)):
        for f in fileNames:
            file_path = os.path.join(dirPath, f) #取圖片路徑
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) #以灰階模式加載圖片
            res=cv2.resize(img,(600,500),interpolation=cv2.INTER_CUBIC)
            data_list.append(res) #轉成灰階圖片矩陣
            #label = dirPath.split('/')[-1]
            label = dirPath.split('\\')[-1] #取資資料夾名稱
            label_list.append(label)
            #label_list.remove("./training")
    return (np.asarray(data_list, dtype=np.float32), np.asarray(label_list))

# load dataset
x_dataset, y_dataset = read_dataset("./train") # X:圖片灰階矩陣,Y:Label 0~9

# Import train_test_split
from sklearn.cross_validation import train_test_split

# Split the data into training and testing sets with 20% test rate
X_train, X_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size = 0.2, random_state = 0)

#Take a look at a iceberg
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)

def plotmy3d(c, name):
    data = [go.Surface(z=c)]
    layout = go.Layout(title=name,autosize=False,width=700,height=700,margin=dict(l=65,r=10,b=15,t=90))
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)
    
plotmy3d(arr1, 'Defect')

# Data Preprocess
def get_scaled_imgs(df):
    imgs = []
    for i, row in df.iterrows():
        img1 = np.array(row['im1']).reshape(500, 600)
        # Normalize
        img2 = (img1 - img1.mean()) / (img1.max() - img1.min())
        imgs.append(np.dstack((img2)))
    return np.array(imgs)

# Reshape data for SVD
im1 = np.reshape(X_train,(50, 300000))
U1,s1,V1 = np.linalg.svd(im1,full_matrices = 0)
plt.imshow(np.reshape(im1[4,:],(500,600)))

#Orignal Picture

fig, ax = plt.subplots(2,3)
plt.suptitle('Original Picture')
ax[0,0].imshow(np.reshape(im1[0,:],(500,600)))
ax[0,1].imshow(np.reshape(im1[1,:],(500,600)))
ax[0,2].imshow(np.reshape(im1[2,:],(500,600)))
ax[1,0].imshow(np.reshape(im1[3,:],(500,600)))
ax[1,1].imshow(np.reshape(im1[4,:],(500,600)))
ax[1,2].imshow(np.reshape(im1[5,:],(500,600)))

#one-hot encoding
num_classes = 6

pattern = {'MA':'0','PA':'1','RE':'2','DC':'3','PC':'4','RU':'5'}
y_train = [pattern[x] if x in pattern else x for x in y_train]
y_test = [pattern[x] if x in pattern else x for x in y_test]

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#reshape numpy array for keras
(mnist_row, mnist_col, mnist_color) = 500, 600, 1

X_train = X_train.reshape(X_train.shape[0], mnist_row, mnist_col, mnist_color) #(50, 500, 600) -> (50, 500, 600, 1)
X_test = X_test.reshape(X_test.shape[0], mnist_row, mnist_col, mnist_color) #(13, 500, 600) -> (13, 500, 600, 1)

#feature scaling
X_train /= 255
X_test /= 255

#create your keras model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(mnist_row, mnist_col, mnist_color) ))
model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

model.summary()

train_history=model.fit(x_train,y_train, validation_data=(x_train,y_train), validation_split=0.2, epochs=30, batch_size=10, verbose=1)

# save model
model.save('model_OM.h5')

scores = model.evaluate(x_test,y_test)
scores[1]

# train/validation result
import matplotlib.pyplot as plt
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train_History')
    plt.ylabel('train')
    plt.xlabel('Epoch')
    plt.legend(['train','validation'], loc='center right')
    plt.show()
    
show_train_history(train_history,'acc','val_acc')

# test result (confuse matrix)
import pandas as pd
prediction = model.predict_classes(x_test)
y_test2 = y_dataset[test_list]
print(y_test2.shape)
pd.crosstab(y_test2, prediction, rownames=['label'], colnames=['predict'])

