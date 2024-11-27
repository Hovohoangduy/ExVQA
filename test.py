import torch
from PIL import Image
import torchvision.transforms as transforms
from config import config
from transformers import AutoTokenizer
from vqa_model import VQAModel


model = VQAModel().to(config.DEVICE)
model.load_state_dict(torch.load('/Users/duyhoang/Documents/Research/VQAG/code/models/PathVQA_lstm.pt', map_location=config.DEVICE))
tokenizer = AutoTokenizer.from_pretrained(config.TEXT_DIR)


def predict(model, image_path, question, device):
    model.eval() 

    with torch.no_grad():
        image = Image.open(image_path).convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
        ])
        
        image = transform(image).to(device)
        
        
        predicted_tokens = model(image.to(config.DEVICE), question)
        sentence_predicted = torch.argmax(predicted_tokens[0], axis=1)
        predicted_sentence = tokenizer.decode(sentence_predicted, skip_special_tokens=True).strip()

    print(f"Question: {question}")
    print(f"Predicted Answer: {predicted_sentence}")
    
    return predicted_sentence

image_path = "/Users/duyhoang/Documents/Research/VQAG/code/sample/test2.jpg" 
question = "is normal palmar creases present?"

predicted_answer = predict(model, image_path, question, device=config.DEVICE)