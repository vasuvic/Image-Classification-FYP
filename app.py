import os
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input 
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import tkinter as tk
from tkinter import filedialog, Label, Button
from tkinter.messagebox import showinfo
from cv2 import imshow, waitKey, destroyAllWindows
from flask import Flask, render_template, request
from keras.models import load_model
#from keras.preprocessing import image
from PIL import Image

app = Flask(__name__)

# Load categories
categories = os.listdir("D:/book_image_data_species_wise_image_classification/Amaryllidaceae/1)dataset_for_model/train")
categories.sort()

# Load the saved model
path_for_saved_model = "model.h5"
model = tf.keras.models.load_model(path_for_saved_model)
model.make_predict_function()

# def classify_image(imageFile):
#     img = Image.open(imageFile)
#     img = img.resize((224, 224), Image.ANTIALIAS)
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)

#     pred = model.predict(x)
#     categoryValue = np.argmax(pred, axis=1)[0]
#     return categories[categoryValue]


def classify_image(imageFile):
    x = []
    
    img = Image.open(imageFile)
    img.load()
    img = img.resize((224,224), Image.Resampling.LANCZOS)  # Updated line
    
    x = image.img_to_array(img)
    x = np.expand_dims(x , axis=0)
    x = preprocess_input(x)
    print(x.shape)
    
    pred = model.predict(x)
    categoryValue = np.argmax(pred , axis = 1)
    print(categoryValue)
    
    categoryValue = categoryValue[0]
    print(categoryValue)
    
    result = categories[categoryValue]
    
    return result

def upload_and_predict():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )
    if file_path:
        result = classify_image(file_path)
        showinfo("Prediction Result", f"The image belongs to the category: {result}")

        # Display image with prediction
        img = cv2.imread(file_path)
        img = cv2.resize(img, (800, 800))  # Resize image for better visibility
        img = cv2.putText(img, result, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        imshow("Prediction Result", img)
        waitKey(0)
        destroyAllWindows()


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		# img_path = "static/" + img.filename	
		img_path = img.filename	
		img.save(img_path)

		p = classify_image(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)


# from flask import Flask, render_template, request
# from keras.models import load_model
# from keras.preprocessing import image

# app = Flask(__name__)

# #dic = {0 : 'Cat', 1 : 'Dog'}
# dic = {0: "Crinum asiaticum", 1:"Crinum latifolium", 2:"Curculigo orchioides", 3:"Narcissus tazetta", 4:"Polianthes tuberosa"}

# model = load_model('model.h5')

# model.make_predict_function()

# def predict_label(img_path):
# 	i = image.load_img(img_path, target_size=(100,100))
# 	i = image.img_to_array(i)/255.0
# 	i = i.reshape(1, 100,100,3)
# 	p = model.predict_classes(i)
# 	return dic[p[0]]


# # routes
# @app.route("/", methods=['GET', 'POST'])
# def main():
# 	return render_template("index.html")

# @app.route("/about")
# def about_page():
# 	return "Please subscribe  Artificial Intelligence Hub..!!!"

# @app.route("/submit", methods = ['GET', 'POST'])
# def get_output():
# 	if request.method == 'POST':
# 		img = request.files['my_image']

# 		img_path = "static/" + img.filename	
# 		img.save(img_path)

# 		p = predict_label(img_path)

# 	return render_template("index.html", prediction = p, img_path = img_path)


# if __name__ =='__main__':
# 	#app.debug = True
# 	app.run(debug = True)