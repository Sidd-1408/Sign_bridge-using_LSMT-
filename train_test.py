import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the root directory of your dataset
root_dir = "Desktop/data"

# Define the list of folders containing your images
folders = ['1','2','3','4','5','6','7','8','9','A',"B",'C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# Initialize empty lists for the images and labels
images = []
labels = []

# Loop through each folder
for folder in folders:
    # Get the path to the current folder
    folder_path = os.path.join(root_dir, folder)
    
    # Loop through each image in the folder
    for filename in os.listdir(folder_path):
        # Get the path to the current image
        img_path = os.path.join(folder_path, filename)
        
        # Load the image using OpenCV
        img = cv2.imread(img_path)
        
        # Preprocess the image (e.g. resize, convert to grayscale, etc.)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize the image to a suitable size
        resized_image = cv2.resize(gray_image, (64, 64))

        # Normalize the pixel values
        normalized_image = resized_image / 255.0
        
        # Add the preprocessed image and its label to the lists
        images.append(normalized_image)
        labels.append(folder)
        
# Convert the image and label lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# Convert labels to numeric values
label_encoder = LabelEncoder()
train_labels_numeric = label_encoder.fit_transform(train_labels)
test_labels_numeric = label_encoder.transform(test_labels)

# Convert labels to one-hot encoding
num_classes = len(label_encoder.classes_)
train_labels_one_hot = to_categorical(train_labels_numeric, num_classes)
test_labels_one_hot = to_categorical(test_labels_numeric, num_classes)

from keras.models import Sequential
from keras.layers import LSTM, Dense

# Define the model architecture
model = Sequential()
model.add(LSTM(units=128, input_shape=(train_data.shape[1], train_data.shape[2])))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=len(folders), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_data, train_labels_one_hot, epochs=50, batch_size=32, validation_data=(test_data, test_labels_one_hot))
# Evaluate the model on the testing set
test_loss, test_acc = model.evaluate(test_data, test_labels_one_hot)

# Print the test loss and accuracy
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
# Make predictions on the testing set
predictions = model.predict(test_data)

# Convert the predictions from one-hot encoding to class labels
predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_data, test_labels_one_hot)

# Print the test loss and accuracy
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


#Testing
# Load a new image
new_image = cv2.imread("Desktop/test5.png")

# Preprocess the image
gray_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
resized_image = cv2.resize(gray_image, (64, 64))
normalized_image = resized_image / 255.0
# Reshape the image to match the input shape of the model
input_image = normalized_image.reshape(1, 64, 64)
# Make a prediction using the trained model
prediction = model.predict(input_image)
# Convert the prediction from one-hot encoding to a class label
predicted_class = label_encoder.inverse_transform(np.argmax(prediction, axis=1))
print("Predicted class:", predicted_class)


# Save the trained model to disk
model.save('proto_model.h5')
