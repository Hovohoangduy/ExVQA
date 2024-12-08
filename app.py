from flask import Flask, render_template, request, jsonify
import os
import torch
from torchvision import transforms
from PIL import Image
from transformers import AutoTokenizer
from vqa_model import VQAModel
from config import config
from test import predict_batch

app = Flask(__name__)
os.makedirs("static/temp", exist_ok=True)

# Load model và tokenizer
model = VQAModel().to(config.DEVICE)
model.load_state_dict(torch.load('static/weights/ExVQA_v2.pt', map_location=config.DEVICE))
tokenizer = AutoTokenizer.from_pretrained(config.TEXT_DIR)

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

@app.route('/predict_batch', methods=['POST'])
def predict_batch_route():
    """
    API nhận đầu vào 1 ảnh và 1 câu hỏi, sau đó copy ảnh thành 4 lần 
    và dự đoán dựa trên danh sách câu hỏi mặc định.
    """
    if 'image' not in request.files or 'question' not in request.form:
        return jsonify({'error': 'No image or question provided!'}), 400

    file = request.files['image']  # Nhận ảnh từ người dùng
    user_question = request.form['question']  # Câu hỏi từ người dùng

    default_questions = [
        "What is the condition of the patient?",
        "What kind of scan is shown in this image?",
        "Is there a fracture visible in the scan?"
    ]

    questions = [user_question] + default_questions

    try:
        temp_path = os.path.join("static/temp", file.filename)
        file.save(temp_path)
        image_paths = [temp_path] * len(questions)
        predicted_answers = predict_batch(model, image_paths, questions, device=config.DEVICE)

        os.remove(temp_path)
        return jsonify({
            'user_question': user_question,
            'questions': questions,
            'predictions': predicted_answers
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

### Chạy ứng dụng Flask
if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)