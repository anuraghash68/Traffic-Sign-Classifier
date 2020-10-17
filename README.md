# Traffic-Sign-Classifier
Build a model that will take image as input and able to recognise the sign of the image.


TRAFFIC SIGNS RECOGNITION SYSTEM USING PYTHON

Outline of the Project
In this Python project, I built a deep neural network model that classified traffic signs present in the image into different categories. With this model, I was able to read and understand traffic signs which are a very important task for all autonomous vehicles.
The Dataset
For this project, I used the public dataset available at Kaggle:
This dataset has more than 50,000 images of different traffic signs which are further classified into 43 different classes. The size of the dataset is around 300 MB. The dataset has a train folder which contains images inside each class and a test folder which I used for testing my model.
https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

Install all the python packages.
here are they,
Numpy,pandas,matplotlib,scikitlearn,PIL.

STEPS INVOLVED IN BUILDING THE PROJECT

I started with the project by downloading and unzip of the file from this link –
https://drive.google.com/file/d/1BGDHe6qQwrBEgnl-tXTSKo6TvDj8U3wS/view

I extracted the files into a folder such that I had a train, test and a meta folder. Then I created a Python script file and named it traffic_signs.py in the project folder.

STEP - 1
Exploring the Dataset

My ‘train’ folder has 43 folders each representing a different class. I indexed the folders from 0 to 42. By using the OS module, I iterated over all the classes and appended images and their respective labels in the data and labels list.

The PIL library is used to open image content into an array.


Finally, I stored all the images and their labels into lists (data and labels).
I converted the list into numpy arrays for feeding to the model.

The shape of data is (39209, 30, 30, 3) which means that there are 39,209 images of size 30×30 pixels and the last 3 means the data contains colored images (RGB value).

With the sklearn package, I used the train_test_split() method to split training and testing data.

From the keras.utils package, I used to_categorical method to convert the labels present in y_train and t_test into one-hot encoding.


.

STEP - 2
Building a CNN model

To classify the images into their respective categories, I built a CNN model (Convolutional Neural Network). CNN is best option for image classification purposes.

The architecture of our model is:
The architecture of our model is:

2 Conv2D layer (filter=32, kernel_size=(5,5), activation=”relu”)
MaxPool2D layer ( pool_size=(2,2))
Dropout layer (rate=0.25)
2 Conv2D layer (filter=64, kernel_size=(3,3), activation=”relu”)
MaxPool2D layer ( pool_size=(2,2))
Dropout layer (rate=0.25)
Flatten layer to squeeze the layers into 1 dimension
Dense Fully connected layer (256 nodes, activation=”relu”)
Dropout layer (rate=0.5)
Dense layer (43 nodes, activation=”softmax”)

I then compiled the model with Adam optimizer which performs well and loss is “categorical_crossentropy” because I had multiple classes to categorize.


.

STEP - 3
Training and validating the model

After building the model architecture, I then trained the model using model.fit(). I tried with batch size 32 and 64. My model performed better with 64 batch size. And after 15 epochs the accuracy was stable.


My model got a 95% accuracy on the training dataset. With matplotlib, I plotted a graph for accuracy and the loss.

I. Plotting Accuracy


II. Accuracy and Loss Graphs


.

STEP - 4
Testing my model with test dataset

My dataset had a test folder and in a test.csv file, it had the details related to the image path and their respective class labels. I extracted the image path and labels using pandas. Then to predict the model, I had to resize my images to 30×30 pixels and make a numpy array containing all image data. From the sklearn.metrics, I imported the accuracy_score and observed how my model predicted the actual labels. Thus I achieved a 95% accuracy in this model.


In the end, I saves the model that I’ve trained using the Keras model.save() function.

model.save(‘traffic_classifier.h5’) 
.

This is how I finally implemented it.

import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt 
import cv2 
import tensorflow as tf 
from PIL import Image 
import os 
from sklearn.model_selection import train_test_split 
from keras.utils import to_categorical 
from keras.models import Sequential, load_model 
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout 
 
data = [] 
labels = [] 
classes = 43 
cur_path = os.getcwd() 
 
#Retrieving the images and their labels  
for i in range(classes): 
  path = os.path.join(cur_path,'train',str(i)) 
  images = os.listdir(path) 
 
  for a in images: 
    try: 
      image = Image.open(path + '\\'+ a) 
      image = image.resize((30,30)) 
      image = np.array(image) 
 	  #sim = Image.fromarray(image) 
 	  data.append(image) 
 	  labels.append(i) 
 	except: 
 	  print("Error loading image") 
 
#Converting lists into numpy arrays 
data = np.array(data) 
labels = np.array(labels) 
 
print(data.shape, labels.shape) 
#Splitting training and testing dataset 
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42) 
 
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape) 
 
#Converting the labels into one hot encoding 
y_train = to_categorical(y_train, 43) 
y_test = to_categorical(y_test, 43) 
 
#Building the model 
model = Sequential() 
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:])) 
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu')) 
model.add(MaxPool2D(pool_size=(2, 2))) 
model.add(Dropout(rate=0.25)) 
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu')) 
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu')) 
model.add(MaxPool2D(pool_size=(2, 2))) 
model.add(Dropout(rate=0.25)) 
model.add(Flatten()) 
model.add(Dense(256, activation='relu')) 
model.add(Dropout(rate=0.5)) 
model.add(Dense(43, activation='softmax')) 
 
#Compilation of the model 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
 
epochs = 15 
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test)) 
model.save("my_model.h5") 
 
#plotting graphs for accuracy  
plt.figure(0) 
plt.plot(history.history['accuracy'], label='training accuracy') 
plt.plot(history.history['val_accuracy'], label='val accuracy') 
plt.title('Accuracy') 
plt.xlabel('epochs') 
plt.ylabel('accuracy') 
plt.legend() 
plt.show() 
 
plt.figure(1) 
plt.plot(history.history['loss'], label='training loss') 
plt.plot(history.history['val_loss'], label='val loss') 
plt.title('Loss') 
plt.xlabel('epochs') 
plt.ylabel('loss') 
plt.legend() 
plt.show() 
 
#testing accuracy on test dataset 
from sklearn.metrics import accuracy_score 
 
y_test = pd.read_csv('Test.csv') 
 
labels = y_test["ClassId"].values 
imgs = y_test["Path"].values 
 
data=[] 
 
for img in imgs: 
   image = Image.open(img) 
   image = image.resize((30,30)) 
   data.append(np.array(image)) 
 
X_test=np.array(data) 
 
pred = model.predict_classes(X_test) 
 
#Accuracy with the test data 
from sklearn.metrics import accuracy_score 
print(accuracy_score(labels, pred)) 
 
model.save(‘traffic_classifier.h5’) 
.

.

Traffic Signs Classifier GUI
Now I will show you how I built a graphical user interface for my traffic signs classifier with Tkinter. Tkinter is a GUI toolkit in the standard python library. I made a new file in the project folder and copied the below code. Saved it as gui.py and executed the code by typing python gui.py in the command line.

In this file, I first loaded the trained model ‘traffic_classifier.h5’ using Keras. And then I built the GUI for uploading the image and used a button to classify which calls the classify() function. The classify() function is converting the image into the dimension of shape (1, 30, 30, 3). This is because to predict the traffic sign I have to provide the same dimension I have used when I built the model. Then I predict the class, the model.predict_classes(image) returns me a number between (0-42) which represents the class it belongs to. I used the dictionary to get the information about the class. Here’s the code for the gui.py file.

CODE:

import tkinter as tk 
from tkinter import filedialog 
from tkinter import * 
from PIL import ImageTk, Image 
 
import numpy 
#load the trained model to classify sign 
from keras.models import load_model 
model = load_model('traffic_classifier.h5') 
 
#dictionary to label all traffic signs class. 
classes = { 1:'Speed limit (20km/h)', 
 			2:'Speed limit (30km/h)',  
 			3:'Speed limit (50km/h)',  
 			4:'Speed limit (60km/h)',  
 			5:'Speed limit (70km/h)',  
 			6:'Speed limit (80km/h)',  
 			7:'End of speed limit (80km/h)',  
 			8:'Speed limit (100km/h)',  
 			9:'Speed limit (120km/h)',  
 			10:'No passing',  
 			11:'No passing veh over 3.5 tons',  
 			12:'Right-of-way at intersection',  
 			13:'Priority road',  
 			14:'Yield',  
 			15:'Stop',  
 			16:'No vehicles',  
 			17:'Veh > 3.5 tons prohibited',  
 			18:'No entry',  
 			19:'General caution',  
 			20:'Dangerous curve left',  
 			21:'Dangerous curve right',  
 			22:'Double curve',  
 			23:'Bumpy road',  
 			24:'Slippery road',  
 			25:'Road narrows on the right',  
 			26:'Road work',  
 			27:'Traffic signals',  
 			28:'Pedestrians',  
			29:'Children crossing',  
 			30:'Bicycles crossing',  
 			31:'Beware of ice/snow', 
 			32:'Wild animals crossing',  
 			33:'End speed + passing limits',  
 			34:'Turn right ahead',  
 			35:'Turn left ahead',  
 			36:'Ahead only',  
 			37:'Go straight or right',  
 			38:'Go straight or left',  
 			39:'Keep right',  
 			40:'Keep left',  
 			41:'Roundabout mandatory',  
 			42:'End of no passing',  
 			43:'End no passing veh > 3.5 tons' } 
 
#initialise GUI 
top=tk.Tk() 
top.geometry('800x600') 
top.title('Traffic sign classification') 
top.configure(background='#CDCDCD') 
 
label=Label(top,background='#CDCDCD', font=('arial',15,'bold')) 
sign_image = Label(top) 
 
def classify(file_path): 
 	global label_packed 
 	image = Image.open(file_path) 
 	image = image.resize((30,30)) 
 	image = numpy.expand_dims(image, axis=0) 
 	image = numpy.array(image) 
 	pred = model.predict_classes([image])[0] 
 	sign = classes[pred+1] 
 	print(sign) 
 	label.configure(foreground='#011638', text=sign)  
 
def show_classify_button(file_path): 
 	classify_b=Button(top,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5) 
 	classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold')) 
 	classify_b.place(relx=0.79,rely=0.46) 
 
def upload_image(): 
 	try: 
 		file_path=filedialog.askopenfilename() 
 		uploaded=Image.open(file_path) 
 		uploaded.thumbnail(((top.winfo_width()/2.25),		(top.winfo_height()/2.25))) 
 		im=ImageTk.PhotoImage(uploaded) 
 
 		sign_image.configure(image=im) 
 		sign_image.image=im 
 		label.configure(text='') 
 		show_classify_button(file_path) 
 	except: 
 		pass 
 
upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5) 
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold')) 
 
upload.pack(side=BOTTOM,pady=50) 
sign_image.pack(side=BOTTOM,expand=True) 
label.pack(side=BOTTOM,expand=True) 
heading = Label(top, text="Know Your Traffic Sign",pady=20, font=('arial',20,'bold')) 
heading.configure(background='#CDCDCD',foreground='#364156') 
heading.pack() 
top.mainloop() 
