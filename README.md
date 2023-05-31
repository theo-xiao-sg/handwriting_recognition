# Recognition of Handwriting Numbers

This project is an AI agent trained using KNN and Neural Network model to recognize handwriting numbers. The numbers are drawn in real-time by holding down the left mouse button in the Pygame window. 

## Running Guide

This project is based on the Python programming language and primarily utilizes standard libraries like Pillow, scikit-learn, joblib, pgzero, and pygame.

### Environment Setup

Download the requirements.txt and install the required Python libraries. Please note all my 4 projects share the same requirements.txt. If you have done the installation for one project, you can skip it for the other 3 projects

```bash
# install all packages using requirements.txt
python -m pip install -r requirements.txt
```

### Training the Model

* If you want to train your model, you can run `KNN_model.py` or `neural_network_model.py` in the folder. The model will train itself using 70% of the total 1447 images in the folder `images_training`. I have selected the best model parameters for you in the codes but you can always try yourselves with a set of parameters. For `KNN_model.py`, you can amend the line 29. For `neural_network_model.py`, you can amend the line 31.
* Then, the trained model file saved in a pickle file, either `handwriting_knn.pkl` or `handwriting_knn.pkl`, will be generated in the folder.

### Try the recognition tool

* Run `testing.py` in the folder, a pygame window will show up. In the line 8 of `testing.py`, I have chosen the model which I like the most and it is a Neural Network model. If you like, you can try the other KNN model or any model you trained by amending the line 7 or 8 of `testing.py`.
* Hold down the left mouse button and draw a number from 0 to 9, and click the key `Detect`, then you get the recognized number using the AI model you just trained. I hope it is a correct recognition. For me, most of the tests came with correct answers. 
* Then, please click the key `Clear` to clear the canvas before you try the next number.
* Try all the numbers you like.
* I hope you are impressed by this tool by now!

## Results Illustration

* I have made an animated image file `handwriting_recog_demo.gif` and a video file `handwriting_recog_demo.mp4` to illustrate the number recognition tool. Both files are saved in the folder `results`.
* You can go to the folder `results` and simply click the file `handwriting_recog_demo.gif` in Github. You can immediately see how powerful this number recognition tool is.
* If you download everything and use windows pc, you can open `handwriting_recog_demo.mp4` by Media Player, or most browsers, including Chrome and Edge.
* If you download everything and use a Windows pc, you can open `handwriting_recog_demo.gif` by Photos, or most browsers, including Chrome and Edge. You should not use Paint since it only opens the 1st image.
* I hope you are impressed by this tool as I was!
