from __future__ import division, print_function

import os
import numpy as np
import shutil
import datetime
import random
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image, ImageOps

import datetime
from tensorflow.keras.models import load_model

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/brain_Tumor10.h5'
MODEL_PATH2 = 'models/keras_model.h5'
# Load your trained model
session = tf.Session(graph=tf.Graph())
with session.graph.as_default():
    tf.keras.backend.set_session(session)
    model = tf.keras.models.load_model(MODEL_PATH)
    model2= tf.keras.models.load_model(MODEL_PATH2)
    model._make_predict_function()
    model2._make_predict_function()

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path,model):
    img = load_img(img_path, grayscale=True)
    img_array = img_to_array(img)
    img_array = np.reshape(img, (256, 256, 1))
    with session.graph.as_default():
        tf.keras.backend.set_session(session)
        result = model.predict(img_array.reshape(1, 256, 256, 1))
        result = (result > .5).astype(np.uint8)
    return img_array,result

def classify_(img_path,model2):
    np.set_printoptions(suppress=True)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(img_path)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    image_array = image_array[..., :3]
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    with session.graph.as_default():
        tf.keras.backend.set_session(session)
        prediction = model2.predict(data)
    res_val = np.argmax(prediction)
    return res_val

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        req=request.form['task_num']
        req=int(req)
        f = request.files['file']
        shutil.rmtree('uploads/')
        os.mkdir("uploads")
        shutil.rmtree('static\\uploads')
        os.mkdir("static\\uploads")
        basepath = os.path.dirname(__file__)
        filename = secure_filename(f.filename)
        file_path = os.path.join(
             basepath, "uploads", secure_filename(f.filename)
        )
        f.save(file_path)
        if req==1:
            print(os.path.join(basepath,"static\\uploads",filename))
            current_time = datetime.datetime.now()
            str1 = str(current_time.hour) + str(current_time.minute) + str(current_time.second) + str(
                current_time.microsecond)
            shutil.copy(file_path, os.path.join(basepath, "static\\uploads", str1+filename))

            pred=classify_(file_path,model2)
            if pred==0:
                to_send="Non"
            else:
                to_send="Oui"
            fname = str1+secure_filename(f.filename)
            return render_template('classify_result.html',fname=fname,c_result=to_send)
        else:
            # Make prediction
            data,preds=model_predict(file_path, model)
            plt.axis('off')
            plt.imshow(np.squeeze(data,axis=-1), cmap='gray')
            preds = np.ma.masked_where(preds == False, preds)
            plt.imshow(np.squeeze(preds,axis=-1).reshape(256,256), alpha=0.6, cmap='Set1')
            shutil.rmtree('static/result/')
            os.mkdir('static/result')
            current_time = datetime.datetime.now()
            str1 = str(current_time.hour) + str(current_time.minute) + str(current_time.second) + str(
                current_time.microsecond)
            fname=str(str1)+secure_filename(f.filename)
            plt.savefig('static/result/'+fname,bbox_inches='tight',transparent=True, pad_inches=0)
            print("here")
            return render_template('result.html',fname=fname)



if __name__ == '__main__':
    app.run(host="0.0.0.0")

