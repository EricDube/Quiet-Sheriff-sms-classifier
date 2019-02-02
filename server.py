from fastai.basic_train import load_callback
from flask import Flask, request, jsonify

import time

from fastai.text import *
import fastai

from settings import *

path = Path(data_dir)

fastai.device = torch.device('cpu')  # run inference on cpu

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

    return jsonify({
        "prediction": str(pred_class),
        "accuracy": str(outputs)
    })


def load_cpu_learner(path:PathOrStr, fname:PathOrStr='export.pkl', test:ItemList=None):
    state = torch.load(open(Path(path)/fname, 'rb'), map_location='cpu')
    model = state.pop('model')
    src = LabelLists.load_state(path, state.pop('data'))
    if test is not None: src.add_test(test)
    data = src.databunch()
    cb_state = state.pop('cb_state')
    clas_func = state.pop('cls')
    res = clas_func(data, model, **state)
    res.callback_fns = state['callback_fns'] #to avoid duplicates
    res.callbacks = [load_callback(c,s, res) for c,s in cb_state.items()]
    return res


if __name__ == '__main__':
    learn = load_cpu_learner(path)

    app.run(host="0.0.0.0", debug=True, port=PORT)


