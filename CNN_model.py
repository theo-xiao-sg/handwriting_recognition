# cnn model

from matplotlib import pyplot as plt
import joblib, os
from PIL import Image
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping


import tensorflow as tf
tf.keras.utils.set_random_seed(1)

# to account for multiple times of MaxPooling2D of 2X2
image_size = 32

# load image dataset
def get_data_my_images():
    X_data = []
    Y_data = []
    for img_name in os.listdir('images_training'):
        imgpath = 'images_training/' + img_name
        label = img_name[0]
        image = Image.open(imgpath).convert('L')
        image = image.resize((image_size, image_size))
        image_array = np.array(image).astype('float32')
        image_array = image_array / 255.0
        image_array = image_array.reshape((image_size, image_size, 1))
        X_data.append(image_array)
        Y_data.append(label)

    return X_data, Y_data

# define cnn model without much of hyperparameter tuning
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 1), padding='same'))
	model.add(MaxPooling2D((2)))

	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D((2)))

	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D((2)))

	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D((2)))

	model.add(Flatten())

	model.add(Dense(10, activation='softmax'))

	# compile model
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model



def plot_learning_progress(history):

	# plot accuracy
	plt.title('Accuracy')
	plt.plot(history.history['accuracy'], color='blue', label='train')
	plt.plot(history.history['val_accuracy'], color='orange', label='validation')
	plt.xlabel('Epoch')
	plt.xticks(range(0, len(history.history['loss']), 2))
	plt.xticks(rotation=45, ha='right')
	plt.ylabel('Accuracy')
	plt.tight_layout()
	plt.grid(True)
	plt.legend()

	# save plot to file
	plt.show(block=False)
	plt.savefig('learning_process_4layer_cnn.jpg')
	plt.close()

# load dataset
X_data, Y_data = get_data_my_images()
X_data = np.array(X_data)

# one hot encode Y_data
y_data_cat = to_categorical(Y_data)

# split data into training set and testing set
x_train, x_test, y_train, y_test = train_test_split(X_data, y_data_cat, test_size=0.3, stratify=y_data_cat)

# define model
model = define_model()

# define early stopping
early_stopping_monitor = EarlyStopping(patience=5, monitor='val_accuracy')

# fit model
history = model.fit(x_train, y_train, epochs=20, validation_split=0.2,
					callbacks=[early_stopping_monitor])

# evaluate model using the best model parameters saved
print('\n')
model.evaluate(x_test, y_test)

# learning curves
plot_learning_progress(history)

print('\n')
model.summary()

# save model
joblib.dump(model,"cnn.pkl")