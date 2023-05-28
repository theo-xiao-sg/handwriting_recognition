from PIL import Image
import numpy as np
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def get_data():
    data = {'features': [], 'labels': []}
    for img_name in os.listdir('images_training'):
        imgpath = 'images_training/' + img_name
        label = img_name[0]
        img = Image.open(imgpath).convert('1')
        img = img.resize((50, 50))
        img_num = np.array(img).reshape(-1)
        img_num = list(img_num)
        data['features'].append(img_num)
        data['labels'].append(int(label))

    return data


data = get_data()
# split data into training set and testing set
x_train, x_test, y_train, y_test = train_test_split(data['features'], data['labels'], test_size=0.3, random_state=1)

# tried different values of number_of_neurons for one hidden layer
# save number_of_neurons and accuracy in a dictionary
result = {}
# number_of_neurons_list = [10, 50, 100, 300, 500, 700, 900, 1100]
number_of_neurons_list = [700]

for number_of_neurons_i in number_of_neurons_list:
    # create MLP model for each number_of_neurons
    model = MLPClassifier(hidden_layer_sizes=(number_of_neurons_i,))
    print('model training for number_of_neurons_i: {} ...'.format(number_of_neurons_i))

    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)
    accuracy_i = accuracy_score(y_predict, y_test)
    result[number_of_neurons_i] = accuracy_i
    print('number_of_neurons_i: {}, accuracy: {:.2%}'.format(number_of_neurons_i, accuracy_i))



# save model
joblib.dump(model,"handwriting_NeuralNetwork.pkl")
