import torch
import random
import numpy as np
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import torch.optim as optim
from vqa_model import VQAModel
from config import config
from data_processing import train_loader
from features_extraction import AnsEmbedding
from transformers import AutoTokenizer
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained(config.TEXT_DIR)
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model = VQAModel().to(config.DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=2e-4)
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=0, 
    num_training_steps=int(len(train_loader) * 20)
)

ans_model = AnsEmbedding().to(config.DEVICE)
for batch in train_loader:
    images, questions, answers = batch
    predicted_tokens = model(images.to(config.DEVICE), questions)
    predicted_ans = torch.softmax(predicted_tokens, dim=1)
    print(predicted_ans)
    predicted_tokens = predicted_tokens.float()
    answer_indices, ans_embedds = ans_model(answers)
    
    
    sentence_predicted = torch.argmax(predicted_ans[0], axis=1)
    predicted_sentence = tokenizer.decode(sentence_predicted, skip_special_tokens=True).strip()
    # Tính loss
    loss = criterion(predicted_tokens.permute(0, 2, 1), answer_indices)
    loss = criterion(predicted_tokens, ans_embedds)
    print(f"loss: {loss: .4f}")
    break