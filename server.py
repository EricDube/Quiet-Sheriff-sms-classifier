from flask import Flask, request, jsonify

import time

from fastai.text import *
import fastai

from settings import *

path = Path(data_dir)

fastai.device = torch.device('cpu')  # run inference on cpu
data_clas = TextClasDataBunch.from_csv(path, 'train/t200.csv', bs=32)
learn = text_classifier_learner(data_clas, drop_mult=0.5)
learn = learn.load('val-codes-000')

app = Flask(__name__)


@app.route("/")
def hello():
    return "Image classification example\n"


@app.route('/predict', methods=['GET'])
def predict():
    sms = request.args['sms']
    app.logger.info(f"Classifying sms {sms}")

    t = time.time()  # get execution time
    pred_class, pred_idx, outputs = learn.predict(sms)
    dt = time.time() - t

    app.logger.info("Execution time: %0.02f seconds" % (dt))
    app.logger.info("Image %s classified as %s" % (sms, pred_class))

    return jsonify({"prediction": str(pred_class)})


if __name__ == '__main__':
    # app.run(host="0.0.0.0", debug=True)
    app.run(host="0.0.0.0", debug=True, port=PORT)
