import torch
from PIL import Image
import torchvision.transforms as transforms
from config import config
from transformers import AutoTokenizer
from vqa_model import VQAModel
import torch.nn.functional as F

# Load model và tokenizer
model = VQAModel().to(config.DEVICE)
model.load_state_dict(torch.load('/Users/duyhoang/Downloads/med_vqa.pt', map_location=config.DEVICE))
tokenizer = AutoTokenizer.from_pretrained(config.TEXT_DIR)

def predict_batch(model, image_paths, questions, device):
    model.eval()

    with torch.no_grad():
        transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
        ])

        images = [transform(Image.open(image_path).convert("RGB")) for image_path in image_paths]
        images = torch.stack(images).to(device)  # Stack thành batch

        predicted_tokens = model(images, questions)
        predicted_sentences = []

        for i in range(len(questions)):
            sentence_predicted = torch.argmax(predicted_tokens[i], dim=-1)
            predicted_sentence = tokenizer.decode(sentence_predicted, skip_special_tokens=True).strip()
            predicted_sentences.append(predicted_sentence)

    for i in range(len(questions)):
        print(f"Image {i+1}: {image_paths[i]}")
        print(f"Question: {questions[i]}")
        print(f"Predicted Answer: {predicted_sentences[i]}")
        print("-" * 50)
    
    return predicted_sentences

# image = "/Users/duyhoang/Documents/Research/VQAG/code/sample/synpic40127.jpg"
# question = "What is the MR weighting in this image?"

# image_paths = [image, image, image, image]
# questions = [
#     question,
#     "What is the condition of the patient?",
#     "What kind of scan is shown in this image?",
#     "Is there a fracture visible in the scan?"
# ]

# predicted_answers = predict_batch(model, image_paths, questions, device=config.DEVICE)