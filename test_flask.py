import os

import numpy as np
from PIL import Image
from flask import Flask, request, render_template, json
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


@app.route('/postImage', methods=['POST'])
@cross_origin()
def post_image():
    if request.method == 'POST':
        f = request.files['file']
        f.save('images/' + f.filename)
        image_path = 'images/' + f.filename
        if os.path.isfile(image_path):
            im = Image.open(image_path)
            t, e = os.path.splitext(image_path)
            im_resize = im.resize((64, 64), Image.ANTIALIAS).convert('RGB')
            model = load_model("models/mpl100epochs/model.keras.keras")
            result = imageClassification(im_resize, model)
            print(result)
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


# @app.route('/success', methods=['POST'])
# def success():
#     if request.method == 'POST':
#         f = request.files['file']
#         f.save('images/' + f.filename)
#         image_path = 'images/' + f.filename
#         if os.path.isfile(image_path):
#             print(image_path)
#             im = Image.open(image_path)
#             t, e = os.path.splitext(image_path)
#             imResize = im.resize((64,64), Image.ANTIALIAS).convert('RGB')
#             # imResize.save(t + '_resized.jpg', 'JPEG', quality=90)
#
#             model = load_model("models/mpl100epochs/model.keras.keras")
#
#             result = imageClassification(imResize, model)
#             print(result)
#
#         return render_template("successimage.html", name=f.filename)



if __name__ == "__main__":
    app.run(debug=True)

