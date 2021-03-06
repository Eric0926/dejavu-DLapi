from flask import Flask, request, jsonify
from search import *
from config import *
import os

app = Flask(__name__)


@app.route("/search")
def search():
    img_url = request.form["img_url"]
    results_num = int(request.form["results_num"])
    results = None
    # images = vgg_search('https://storage.googleapis.com/flagged_evaluation_images/444_10_r2.png', 10)
    try:
        results = vgg_search(img_url, results_num)
    except:
        return "Invalid image url", 400
    return jsonify(results), 200
    # return "it worked!", 200


if __name__ == "__main__":
    # os.environ["LD_PRELOAD"] = "~/anaconda3/lib/libmkl_core.so:~/anaconda3/lib/libmkl_sequential.so"
    # export LD_PRELOAD=~/anaconda3/lib/libmkl_core.so:~/anaconda3/lib/libmkl_sequential.so
    # export LD_PRELOAD=/opt/conda/lib/libmkl_core.so:/opt/conda/lib/libmkl_sequential.so
    # export LD_PRELOAD=/opt/conda/lib/libmkl_def.so:/opt/conda/lib/libmkl_avx.so:/opt/conda/lib/libmkl_core.so:/opt/conda/lib/libmkl_intel_lp64.so:/opt/conda/lib/libmkl_intel_thread.so:/opt/conda/lib/libiomp5.so
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS
    app.run(host='0.0.0.0', port=8000)
