from flask import Flask, request
from search import *

app = Flask(__name__)


@app.route("/api_predict")
def api_predict():
    images = vgg_search(
        'https://storage.googleapis.com/flagged_evaluation_images/444_10_r2.png', 10)
    for image in images:
        print(image)
