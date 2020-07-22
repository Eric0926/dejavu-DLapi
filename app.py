from flask import Flask, request, jsonify
from search import *
from config import *
import os

app = Flask(__name__)


@app.route("/search")
def search():
    images = vgg_search(
        'https://storage.googleapis.com/flagged_evaluation_images/444_10_r2.png', 10)
    return jsonify(images), 200


if __name__ == "__main__":
    # os.environ["LD_PRELOAD"] = "~/anaconda3/lib/libmkl_core.so:~/anaconda3/lib/libmkl_sequential.so"
    # export LD_PRELOAD=~/anaconda3/lib/libmkl_core.so:~/anaconda3/lib/libmkl_sequential.so
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS
    app.run(host='0.0.0.0', port=5000)
