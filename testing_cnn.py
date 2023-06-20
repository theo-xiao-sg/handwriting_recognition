import pgzrun
import pygame
from PIL import Image
import joblib
import numpy as np

# model = joblib.load('handwriting_knn.pkl')
# model = joblib.load('handwriting_NeuralNetwork.pkl')
model = joblib.load('cnn.pkl')

image_size = 32

WIDTH = 800
HEIGHT = 540

notice = Actor('result', (660, 50))
recog = Actor('detect', (660, 360))
clear = Actor('clear', (660, 460))

result = ''

#  draw the screen
def draw():
    screen.draw.line((20, 20), (520, 20), 'brown')
    screen.draw.line((520, 20), (520, 520), 'brown')
    screen.draw.line((520, 520), (20, 520), 'brown')
    screen.draw.line((20, 520), (20, 20), 'brown')

    clear.draw()
    recog.draw()
    notice.draw()
    screen.draw.text(str(result), (600, 100), fontsize=300, color='brown')

# when mouse is clicked
def on_mouse_down(pos):
    global result
    if clear.collidepoint(pos):
        screen.clear()
        result = ''

    if recog.collidepoint(pos):
        pygame.image.save(screen.surface, 'image_to_recognise.png')

        # the same operation as in model training, convert the image to black and white
        img = Image.open('image_to_recognise.png').convert('L')
        new = img.crop((20, 20, 520, 520))
        # resize image to the same size as the resized training image
        img = new.resize((image_size, image_size))
        image_array = np.array(img).astype('float32')
        image_array = image_array / 255.0
        # reshape the image to 1*32*32*1 to match the training data
        image_array = image_array.reshape((1, image_size, image_size, 1))


        # predict the number
        pre = model.predict(image_array)
        result = np.argmax(pre)


# draw the number when mouse is moving
def on_mouse_move(pos, buttons):
    if mouse.LEFT in buttons:
        screen.draw.filled_circle(pos, 20, 'white')

pgzrun.go()