from flask import Flask, request, jsonify

import time

from fastai.text import *
import fastai

path = Path("data")

fastai.device = torch.device('cpu')  # run inference on cpu

app = Flask(__name__)


@app.route("/")
def hello():
    return app.send_static_file("landing.html")


@app.route('/predict', methods=['GET'])
def predict():
    sms = request.args['sms']

    t = time.time()  # get execution time
    pred_class, pred_idx, outputs = learn.predict(sms)
    dt = time.time() - t

    app.logger.info("Execution time: %0.02f seconds" % (dt))

    max_val = max(outputs.tolist())

    return jsonify({
        "prediction": str(pred_class),
        "accuracy": max_val
    })


if __name__ == '__main__':
    learn = load_learner(path)

    app.run(host="0.0.0.0", debug=True, port=80)
