# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.preprocessing import image
#
# # Define constants
# IMAGE_SIZE = 128
# BATCH_SIZE = 32
# EPOCHS = 20
#
# # Define data directories
# data_dir = r'C:\Chordz\sugarcane\sugarcane RA'
#
# # Data preprocessing and augmentation
# datagen = ImageDataGenerator(
#     rescale=1. / 255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     validation_split=0.2  # Splitting the data into training and validation sets
# )
#
# # Generate training data
# train_generator = datagen.flow_from_directory(
#     data_dir,
#     target_size=(IMAGE_SIZE, IMAGE_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='training'  # Use training subset
# )
#
# # Generate validation data
# validation_generator = datagen.flow_from_directory(
#     data_dir,
#     target_size=(IMAGE_SIZE, IMAGE_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='validation'  # Use validation subset
# )
#
# # Build CNN model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dropout(0.5),
#     Dense(512, activation='relu'),
#     Dense(len(train_generator.class_indices), activation='softmax')
# ])
#
# # Compile the model
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# # Train the model
# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // BATCH_SIZE,
#     epochs=EPOCHS,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // BATCH_SIZE
# )
#
# # Save the trained model
# model.save('sugarcane_disease_model.h5')
#
#
# # Function to make predictions
# def predict_image(model, image_path):
#     img = image.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.
#
#     prediction = model.predict(img_array)
#     predicted_class = train_generator.class_indices
#     inv_predicted_class = {v: k for k, v in predicted_class.items()}
#     predicted_label = inv_predicted_class[np.argmax(prediction)]
#     confidence = np.max(prediction)
#
#     return predicted_label, confidence
#
#
# # Example usage for prediction
# image_path = r'D:\testingCRAFT\aphid.jpeg'  # Replace 'path_to_image' with the path to your image
# predicted_label, confidence = predict_image(model, image_path)
# print(f"Predicted class: {predicted_label}, Confidence: {confidence}")


#Above code is to train my model and after 20 epochs it gives 76% accuracy .Now instead of running it again I am using following code to only do predictions

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
IMAGE_SIZE= 128
# Load the trained model
model = load_model('sugarcane_disease_model.h5')
class_names = ['Bacterial Blight', 'Healthy', 'Red Rot', 'Lokrimava Aphids']


# Function to preprocess an image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    return img_array


# Function to make predictions
def predict_image(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    confidence = np.max(prediction)
    return predicted_class, confidence

# Example usages
image_path = r'D:\testingCRAFT\bacterialblight.jpg'  # Replace 'path_to_image' with the path to your image
predicted_class, confidence = predict_image(image_path)
print(f"Predicted class: {predicted_class}, Confidence: {confidence}")
