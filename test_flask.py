import os

import numpy as np
from PIL import Image
from flask import Flask, request, render_template, json, send_file
from matplotlib import image
from tensorflow.python.keras.models import load_model
from flask_cors import CORS, cross_origin

from app.image_tools.resizer import image_resize
from clean_detection import imageClassification

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/postImage/<string:model>', methods=['POST'])
@cross_origin()
def post_image(model):
    if request.method == 'POST':
        f = request.files['file']
        f.save('images/' + f.filename)
        image_path = 'images/' + f.filename
        if os.path.isfile(image_path):
            im = Image.open(image_path)
            t, e = os.path.splitext(image_path)
            im_resize = im.resize((64, 64), Image.ANTIALIAS).convert('RGB')
            str = "models/%s/model.keras.keras" %model
            model = load_model(str)
            result = imageClassification(im_resize, model)
            response = app.response_class(
                response=json.dumps(result),
                status=200,
                mimetype='application/json'
            )
            return response


@app.route('/getResults', methods=['GET'])
def get_results():
    if request.method == 'GET':
        print('GET RESULTS')


@app.route('/selectModel', methods=['GET'])
@cross_origin()
def selectModel():
    if request.method == 'GET':
        filenames = os.listdir("./models")  # get all files' and folders' names in the current directory
        modelList = {}
        count = 0;
        for filename in filenames:  # loop through all the files and folders
            if os.path.isdir(
                    os.path.join(os.path.abspath("./models"),
                                 filename)):  # check whether the current object is a folder or not
                modelList[count] = filename
                count+=1
        return modelList


@app.route('/displayAccuracy/<string:model>', methods=['GET'])
@cross_origin()
def displayAccuracy(model):
    if request.method == 'GET':
        path = "models/%s/curve/accuracy.png" % (model)
        return send_file(path, mimetype='image/PNG')


@app.route('/displayLoss/<string:model>', methods=['GET'])
@cross_origin()
def displayLoss(model):
    if request.method == 'GET':
        path = "models/%s/curve/loss.png" % (model)
        return send_file(path, mimetype='image/PNG')


if __name__ == "__main__":
    app.run(debug=True)

