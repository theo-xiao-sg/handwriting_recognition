# recognition of handwriting digits by using mouse click

This project is an AI agent trained using KNN and Neural Network model to recognize handwriting digits. The numbers are written by mouse left click in the pygame window. 

## Running Guide

This project is based on the Python programming language and primarily utilizes standard libraries like Pillow, scikit-learn, joblib, pgzero, and pygame.

### Environment Setup

download the requirements.txt and install required Python libraries. please note all my 4 projects share the same requirements.txt. if you have done the installation for one project, you can skip it for the other 3 projects

```bash
# install all packages using requirements.txt
python -m pip install -r requirements.txt
```

### Training the Model

* If you want to train your own model, you can run `KNN_model.py` or `neural_network_model.py` in the folder. The model will train itself using 1447 images in the folder `images_training`. I have selected the best model parameter for you in the codes but you can always try yourselves with a set of parameters.
* Then, the trained model file saved in a pickle file, either `handwriting_knn.pkl` or `handwriting_knn.pkl`, will be generated in the folder.

### Try the recognition tool

* Run `testing.py` in the folder, a pygame window will show up. 
* Use your mouse lift button and draw a number from 0 to 9, and click the key `Detect`, then you get the recognized number using the AI model you just trained. I hope it is a correct recognition. For me, most of the tests came with correct answers. 
* Then, please click the key `Clear` to clear the canvas before you try the next number.
* Try all the numbers you like.
* I hope you are impressed by this tool by now!

## Results Illustration

* I have made an animated images file `handwriting_recog_demo.gif` and a video file `handwriting_recog_demo.mp4` to illustrate the number recognition tool. Both files are saved in the folder `results`.
* You can go to the folder `results` and simply click the file `handwriting_recog_demo.gif` in Github. You can immediately see how powerful this number recognition tool.
* If you download everything and use windows pc, you can open `handwriting_recog_demo.mp4` by Media Player, or most browsers, including Chrome and Edge.
* If you download everything and use windows pc, you can open `handwriting_recog_demo.gif` by Photos, or most browsers, including Chrome and Edge. You should not use Paint since it only opens the 1st image.
* I hope you are impressed by this tool as I was!


