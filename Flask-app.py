import tensorflow as tf
import cv2
import numpy as np
from flask import *

app = Flask("CNN APP")

model = None

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)