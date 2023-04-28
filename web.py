import os
import cv2
import urllib
import uuid
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request, Response, redirect, url_for
from tensorflow.keras.models import load_model
import tempfile
from PIL import Image
app = Flask(__name__)


# Load the trained CNN model
model = load_model('handrecognition_model.h5')
ALLOWED_EXT = set(['jpg','jpeg','png'])
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1) [1] in ALLOWED_EXT
class_names = ["down", "palm", "l", "fist", "fist_moved", "thumb", "index", "ok", "palm_moved", "c"]

def predict(filename, model):

    img = cv2.imread(filename)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converts into the corret colorspace (GRAY)
    img = cv2.resize(img, (320, 120))
    img = np.array(img, dtype="uint8")

    img = img.reshape(1 ,120, 320, 1)

    result = model.predict(img)
    dict_result = {}
    for i in range(10):
        dict_result[result[0][i]] = class_names[i]
    res = result[0]
    res.sort()
    res = res[::-1]
    prob = res[:3]
  
    prob_result = []
    class_result = []
    for i in range(3):
       prob_result.append((prob[i]*100).round(2))
       class_result.append(dict_result[prob[i]])
    return class_result , prob_result



# Initialize the webcam
# cap = cv2.VideoCapture(0)

# Set up the main route
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/success' , methods = ['GET' , 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd() , 'static/images')
    if request.method == 'POST':
        if(request.form):
            link = request.form.get('link')
            try :
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename+".jpg"
                img_path = os.path.join(target_img , filename)
                output = open(img_path , "wb")
                output.write(resource.read())
                output.close()
                img = filename
 
                class_result , prob_result = predict(img_path , model)
 
                predictions = {
                      "class1":class_result[0],
                        "class2":class_result[1],
                        "class3":class_result[2],
                        "prob1": prob_result[0],
                        "prob2": prob_result[1],
                        "prob3": prob_result[2],
                }
 
            except Exception as e :
                print(str(e))
                error = 'This image from this site is not accesible or inappropriate input'   
 
            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = predictions)
            else:
                return render_template('index.html' , error = error)
 
           
        elif (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img , file.filename))
                img_path = os.path.join(target_img , file.filename)
                img = file.filename
 
                class_result , prob_result = predict(img_path , model)
 
                predictions = {
                      "class1":class_result[0],
                        "class2":class_result[1],
                        "class3":class_result[2],
                        "prob1": prob_result[0],
                        "prob2": prob_result[1],
                        "prob3": prob_result[2],
                }
 
            else:
                error = "Please upload right extension"
 
            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = predictions)
            else:
                return render_template('index.html' , error = error)
 
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
