from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
from transformers import AutoTokenizer
from vqa_model import VQAModel
from config import config

app = Flask(__name__)

model = VQAModel().to(config.DEVICE)
model.load_state_dict(torch.load('/Users/duyhoang/Documents/Research/VQAG/code/models/exvqa_model.pt', map_location=config.DEVICE))

tokenizer = AutoTokenizer.from_pretrained(config.TEXT_DIR)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
])

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/model')
def page_model():
    return render_template('model.html')

@app.route('/experiment')
def page_experiment():
    return render_template('experiment.html')

@app.route('/application')
def page_application():
    return render_template('application.html')


def predict(model, image, question, device):
    model.eval() 

    with torch.no_grad():
        image = transform(image).to(device)
        
        predicted_tokens = model(image.to(config.DEVICE), question)
        sentence_predicted = torch.argmax(predicted_tokens[0], axis=1)
        predicted_sentence = tokenizer.decode(sentence_predicted, skip_special_tokens=True).strip()

    return predicted_sentence

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files or 'question' not in request.form:
        return jsonify({'error': 'No image or question provided!'}), 400

    file = request.files['image']
    question = request.form['question']

    try:
        image = Image.open(file).convert('RGB') 
        predicted_answer = predict(model, image, question, device=config.DEVICE)

        return jsonify({'prediction': predicted_answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

### cmt
if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)