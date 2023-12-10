from flask import Flask, jsonify, request
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from flask_cors import CORS, cross_origin

base_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
peft_model = "00BER/imbd-roberta-base-sentiment-latest"
config = PeftConfig.from_pretrained(peft_model)
model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2, ignore_mismatched_sizes=True)
model = PeftModel.from_pretrained(model, peft_model)
tokenizer = AutoTokenizer.from_pretrained(base_model, add_prefix_space=True)
model.eval()

sentiments = ['positive', "negative"]
id2sent = { idx: sentiment for idx, sentiment in enumerate(sorted(sentiments)) }
sent2id = { sentiment: idx for idx, sentiment in enumerate(sorted(sentiments)) }



app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def get_prediction(text):
    tokens = tokenizer(text, return_tensors="pt")
    output = model(**tokens)
    result = output.logits.argmax(-1)
    return id2sent[result.item()]

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        if request.method == 'POST':
            input = request.json["input"]
            prediction = get_prediction(input)
        return jsonify({"prediction": prediction })
