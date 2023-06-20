from PIL import Image
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

from numpy.random import seed
seed(1)

# to align the image size with CNN model
image_size = 32

def get_data():
    data = {'features': [], 'labels': []}
    for img_name in os.listdir('images_training'):
        imgpath = 'images_training/' + img_name
        label = img_name[0]
        img = Image.open(imgpath).convert('L')
        img = img.resize((image_size, image_size))
        img = np.array(img).astype('float32')
        img = img / 255.0
        img_num = np.array(img).reshape(-1)
        img_num = list(img_num)
        data['features'].append(img_num)
        data['labels'].append(int(label))

    return data


data = get_data()
x_train, x_test, y_train, y_test = train_test_split(data['features'], data['labels'], test_size=0.3, stratify=data['labels'])

# save num_neighbors and accuracy in a dictionary
result = {}
# num_neighbors = [1, 3, 5, 7, 9]
num_neighbors = [1]

for num_neighbor_i in num_neighbors:
    # create KNN model for each num_neighbors
    model = KNeighborsClassifier(n_neighbors=num_neighbor_i, weights='distance')
    print('model training for num_neighbors: {} ...'.format(num_neighbor_i))

    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)
    accuracy_i = accuracy_score(y_predict, y_test)
    result[num_neighbor_i] = accuracy_i
    print('num_neighbors: {}, accuracy: {:.2%}'.format(num_neighbor_i, accuracy_i))


# save model
joblib.dump(model,"handwriting_knn.pkl")
