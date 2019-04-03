from flask import Flask, request, jsonify

from fastai.text import *
import fastai

path = Path("data")

fastai.device = torch.device('cpu')  # run inference on cpu

app = Flask(__name__)

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

@app.route('/predict', methods=['GET'])
def predict():
    sms = request.args['sms']

    pred_class, pred_idx, outputs = learn.predict(sms)

    max_val = max(outputs.tolist())

    return jsonify({
        "prediction": str(pred_class),
        "accuracy": max_val
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "healthy": "true"
    })


if __name__ == '__main__':
    learn = load_learner(path)

    app.run(host="0.0.0.0", debug=False, port=8080)
