from src.summarization import summarize
from flask import Flask, request
from flask_cors import CORS, cross_origin
import pickle
import joblib
import os

app = Flask(__name__)
CORS(app)


@app.get('/')
def hello_world():
    return 'Hello, World!'


@app.post('/predict')
def get():
    source = request.get_json()
    sentences = summarize(source)
    text_svm_filepath = os.path.join('src', 'Text_SVM.pkl')
    l = joblib.load(text_svm_filepath)
    count_vect = l[1]
    transformer = l[2]
    model = l[0]
    d = {}
    for sentence in sentences:
        mc = count_vect.transform([sentence])
        m = transformer.transform(mc)
        output = model.predict(m)
        if output[0] not in d.keys():
            newlist = list()
            newlist.append(sentence)
            d[output[0]] = newlist
        else:
            d[output[0]].append(sentence)
    return {'data': d}, 200


if __name__ == '__main__':
    app.run()
