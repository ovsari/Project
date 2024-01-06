#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import seaborn as sns
import sklearn
import os
import shutil
import cv2
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory

#Function to parse through the folders and extract the images from their folders
labels = ['PNEUMONIA', 'NORMAL']
img_size = 224
def get_training_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                resized_arr_rgb_format = cv2.cvtColor(resized_arr, cv2.COLOR_BGR2RGB)
                data.append([resized_arr_rgb_format, class_num])
            except Exception as e:
                print(e)
    return np.array(data)
#Getting the image datasets from paths of the training, test and validation dataset.
train = get_training_data('input/chest-xray-pneumonia/chest_xray/train')
test = get_training_data('input/chest-xray-pneumonia/chest_xray/test')
val = get_training_data('input/chest-xray-pneumonia/chest_xray/val')

#Joining the datasets to enable splitting the dataset using the 80:20 ratio
dataset = np.concatenate((train, val, test), axis=0)
len(dataset)
print(dataset.shape)


#Split the dataset into the training and test dataset
initial_train_df, test_df = train_test_split(dataset, test_size = 0.20, random_state = 30)
#Split the training dataset into the training and val dataset
train_df, val_df = train_test_split(initial_train_df, test_size = 0.20, random_state = 30)
#Function to display the count for the label for each class
def count_labels(labels):
    extracted_labels = [data[1] for data in labels]
    print("Number of labels", len(extracted_labels))
    # Count the occurrences of each class label
    class_counts = np.bincount(extracted_labels)

    # Print the counts
    count_0 = class_counts[0]
    count_1 = class_counts[1]
    print("Count of 0:", count_0)
    print("Count of 1:", count_1)
    sns.set_style('darkgrid')
    sns.countplot(x=extracted_labels).set(title="training data")
#Class distribution for training dataset
count_labels(train_df)


#Seperate the images and labels
x_train = []
y_train = []

x_val = []
y_val = []

x_test = []
y_test = []

for feature, label in train_df:
    x_train.append(feature)
    y_train.append(label)

for feature, label in test_df:
    x_test.append(feature)
    y_test.append(label)
    
for feature, label in val_df:
    x_val.append(feature)
    y_val.append(label)
# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255
x_test = np.array(x_test) / 255
# reshape data for deep learning 
x_train = x_train.reshape(-1, img_size, img_size, 3)
y_train = np.array(y_train)

x_val = x_val.reshape(-1, img_size, img_size, 3)
y_val = np.array(y_val)

x_test = x_test.reshape(-1, img_size, img_size, 3)
y_test = np.array(y_test)
# With data augmentation to prevent overfitting and handling the imbalance in dataset

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)
# from sklearn.utils.class_weight import compute_class_weight
# class_weights = compute_class_weight(class_weight = "balanced", classes= np.unique(train_generator.classes), y= train_generator.classes)
# class_weights = dict(zip(np.unique(train_generator.classes), class_weights))
# class_weights
# Define the early stopping and learning rate reduction callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5, restore_best_weights=True)
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience = 3)


#RESSNET Modeling

#Loading the model
from tensorflow.keras.applications.resnet50 import ResNet50

resnet50_base_model = ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)

#Making sure the layers of the efficientnetv2 model are not retrained 
for layer in resnet50_base_model.layers:
    layer.trainable = False


resnet50_model = tf.keras.models.Sequential()
resnet50_model.add(resnet50_base_model)
resnet50_model.add(tf.keras.layers.Flatten())
resnet50_model.add(tf.keras.layers.BatchNormalization())
resnet50_model.add(tf.keras.layers.Dense(128,activation='relu'))
resnet50_model.add(tf.keras.layers.Dropout(0.5))
resnet50_model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
resnet50_model.summary()

resnet50_model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)
# Train the resnet50 model with early stopping
resnet50_model_history = resnet50_model.fit(
    datagen.flow(x_train,y_train, batch_size = 32),
    epochs=30,
    validation_data=datagen.flow(x_val, y_val),
    callbacks=[early_stopping, learning_rate_reduction])

#Save the model
resnet50_model.save("resnet50_pneumonia_model.h5")

#Plotting the resnet50 model results

#Getting the accuracy
acc = resnet50_model_history.history['accuracy']
val_acc = resnet50_model_history.history['val_accuracy']

#Getting the losses
loss = resnet50_model_history.history['loss']
val_loss = resnet50_model_history.history['val_loss']

#No of epochs it trained
epochs_range = resnet50_model_history.epoch

#Plotting Training and Validation accuracy
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

#Plotting Training and Validation Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


#RESNET50 Performance Evaluation
evaluation_result=resnet50_model.evaluate(x_test,np.array(y_test))
print("Loss of the model is - " , evaluation_result[0])
print("Accuracy of the model is - " , evaluation_result[1]*100 , "%")

resnet50_predictions = resnet50_model.predict(x_test)

y_pred = (resnet50_predictions> 0.5).astype("int32").flatten()
y_pred


y_test

#Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['pneumonia', 'normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)

plt.savefig('resnet50_ConMat24.png')
plt.show()

#Classification report
print(classification_report(y_test, y_pred, target_names = ['Pneumonia (0)','Normal (1)']))

#Accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %f' % (accuracy*100))

#Precision
precision = precision_score(y_test, y_pred)
print('Precision: %f' % (precision*100))


# Recall
recall = recall_score(y_test, y_pred, pos_label=1)
print('Recall: %f' % (recall*100))

#F1-score
F1_score = f1_score(y_test, y_pred)
print('F1_score: %f' % (F1_score*100))


#Specificity 
specificity = recall_score(y_test, y_pred, pos_label=0)
print('Specificity: %f' % (specificity*100))




#Loading the model
from tensorflow.keras.applications.efficientnetv2 import efficientnetv2

efficientnetv2_base_model = efficientnetv2(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)

#Making sure the layers of the efficientnetv2 model are not retrained 
for layer in efficientnetv2_base_model.layers:
    layer.trainable = False

efficientnetv2_model = tf.keras.models.Sequential()
efficientnetv2_model.add(efficientnetv2_base_model)
efficientnetv2_model.add(tf.keras.layers.Flatten())
efficientnetv2_model.add(tf.keras.layers.BatchNormalization())
efficientnetv2_model.add(tf.keras.layers.Dense(128,activation='relu'))
efficientnetv2_model.add(tf.keras.layers.Dropout(0.5))
efficientnetv2_model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
efficientnetv2_model.summary()

efficientnetv2_model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)
# Train the efficientnetv2 model with early stopping
efficientnetv2_model_history = efficientnetv2_model.fit(
    datagen.flow(x_train,y_train, batch_size = 32),
    epochs=30,
    validation_data=datagen.flow(x_val, y_val),
    callbacks=[early_stopping, learning_rate_reduction])


#Plotting the efficientnetv2 model results

#Getting the accuracy
acc = efficientnetv2_model_history.history['accuracy']
val_acc = efficientnetv2_model_history.history['val_accuracy']

#Getting the losses
loss = efficientnetv2_model_history.history['loss']
val_loss = efficientnetv2_model_history.history['val_loss']

#No of epochs it trained
epochs_range = efficientnetv2_model_history.epoch

#Plotting Training and Validation accuracy
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

#Plotting Training and Validation Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


#efficientnetv2 Performance Evaluation¶
evaluation_result=efficientnetv2_model.evaluate(x_test,y_test)
print("Loss of the model is - " , evaluation_result[0])
print("Accuracy of the model is - " , evaluation_result[1]*100 , "%")

efficientnetv2_predictions = efficientnetv2_model.predict(x_test)


y_pred = (efficientnetv2_predictions> 0.5).astype("int32").flatten()
#Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['pneumonia', 'normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)

plt.savefig('efficientnetv2_ConMat24.png')
plt.show()


#Classification report
print(classification_report(y_test, y_pred, target_names = ['Pneumonia (0)','Normal (1)']))


#Accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %f' % (accuracy*100))

#Precision
precision = precision_score(y_test, y_pred)
print('Precision: %f' % (precision*100))

# Recall
recall = recall_score(y_test, y_pred, pos_label=1)
print('Recall: %f' % (recall*100))

#F1-score
F1_score = f1_score(y_test, y_pred)
print('F1_score: %f' % (F1_score*100))

#Specificity 
specificity = recall_score(y_test, y_pred, pos_label=0)
print('Specificity: %f' % (specificity*100))

#Save the model
efficientnetv2_model.save("efficientnetv2_pneumonia_model.h5")



#vit Modeling

#Loading the model
from tensorflow.keras.applications import vitSmall

vit_base_model = vitSmall(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)

#Making sure the layers of the efficientnetv2 model are not retrained 
for layer in vit_base_model.layers:
    layer.trainable = False


vit_model = tf.keras.models.Sequential()
vit_model.add(vit_base_model)
vit_model.add(tf.keras.layers.Flatten())
vit_model.add(tf.keras.layers.BatchNormalization())
vit_model.add(tf.keras.layers.Dense(128,activation='relu'))
vit_model.add(tf.keras.layers.Dropout(0.5))
vit_model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
vit_model.summary()

vit_model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

# Train the resnet50 model with early stopping
vit_model_history = vit_model.fit(
    datagen.flow(x_train,y_train, batch_size = 32),
    epochs=30,
    validation_data=datagen.flow(x_val, y_val),
    callbacks=[early_stopping, learning_rate_reduction])


#Getting the accuracy
acc = vit_model_history.history['accuracy']
val_acc = vit_model_history.history['val_accuracy']

#Getting the losses
loss = vit_model_history.history['loss']
val_loss = vit_model_history.history['val_loss']

#No of epochs it trained
epochs_range = vit_model_history.epoch

#Plotting Training and Validation accuracy
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

#Plotting Training and Validation Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


#vit Model Performance Evaluation¶
evaluation_result=vit_model.evaluate(x_test, y_test)
print("Loss of the model is - " , evaluation_result[0])
print("Accuracy of the model is - " , evaluation_result[1]*100 , "%")
vit_predictions = vit_model.predict(x_test)
y_pred = (vit_predictions> 0.5).astype("int32").flatten()


#Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

import seaborn as sns

#Setting the labels
labels = ['pneumonia', 'normal']

#Plot the Confusion matrix graph
fig= plt.figure(figsize=(8, 5))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')
ax.set_xlabel('Predicted Labels', fontsize=10)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(labels, fontsize = 5)
ax.xaxis.tick_bottom()

ax.set_ylabel('True Labels', fontsize=10)
ax.yaxis.set_ticklabels(labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix', fontsize=15)

plt.savefig('resnet50_ConMat24.png')
plt.show()


#Classification report
print(classification_report(y_test, y_pred, target_names = ['Pneumonia (0)','Normal (1)']))

#Accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %f' %(accuracy*100))

#Precision
precision = precision_score(y_test, y_pred)
print('Precision: %f' % (precision*100))

# Recall
recall = recall_score(y_test, y_pred, pos_label=1)
print('Recall: %f' % (recall*100))

#F1-score
F1_score = f1_score(y_test, y_pred)
print('F1_score: %f' % (F1_score*100))

#Specificity 
specificity = recall_score(y_test, y_pred, pos_label=0)
print('Specificity: %f' % (specificity*100))

#Save the model
vit_model.save("vit_pneumonia_model.h5")
