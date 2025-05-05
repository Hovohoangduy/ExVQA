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
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, brevity_penalty
import math

def train_model(model, ans_model, train_loader, tokenizer, criterion, optimizer, device, num_epochs=14, print_every=500, batch_size=4):
    smoother = SmoothingFunction()

    losses = []
    bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores, bleu_scores = [], [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_bleu_1, total_bleu_2, total_bleu_3, total_bleu_4, total_bleu_scores = 0.0, 0.0, 0.0, 0.0, 0.0

        for batch_idx, batch in enumerate(train_loader):
            images, questions, answers = batch
            if len(images) == batch_size:
                images, questions = images.to(device), questions.to(device)

                predicted_tokens = model(images, questions).float()
                ans_embedds = ans_model(answers)

                references = [answer.strip() for answer in answers]
                hypotheses = []

                for i in range(batch_size):
                    sentence_predicted = torch.argmax(predicted_tokens[i], axis=1)
                    predicted_sentence = tokenizer.decode(sentence_predicted, skip_special_tokens=True)
                    hypotheses.append(predicted_sentence.strip())

                # BLEU score calculations
                bleu_score_1 = corpus_bleu([[ref] for ref in references], hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoother.method1)
                bleu_score_2 = corpus_bleu([[ref] for ref in references], hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother.method1)
                bleu_score_3 = corpus_bleu([[ref] for ref in references], hypotheses, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoother.method1)
                bleu_score_4 = corpus_bleu([[ref] for ref in references], hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother.method1)

                c, r = sum(len(hypothesis) for hypothesis in hypotheses), sum(len(reference) for reference in references)
                bp = brevity_penalty(c, r)
                bleu_scores_all = [bleu_score_1, bleu_score_2, bleu_score_3, bleu_score_4]
                valid_bleu_scores = [score for score in bleu_scores_all if score > 0]
                bleu_score_total = bp * math.exp(sum(math.log(score) for score in valid_bleu_scores) / len(valid_bleu_scores)) if valid_bleu_scores else 0

                total_bleu_1 += bleu_score_1
                total_bleu_2 += bleu_score_2
                total_bleu_3 += bleu_score_3
                total_bleu_4 += bleu_score_4
                total_bleu_scores += bleu_score_total

                # Loss calculation
                loss = criterion(predicted_tokens.view(-1, 50257), ans_embedds.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Log periodically
                if (batch_idx + 1) % print_every == 0:
                    avg_loss_so_far = total_loss / (batch_idx + 1)
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Avg Loss So Far: {avg_loss_so_far:.4f}")
                    print(f"BLEU@1: {bleu_score_1:.4f}, BLEU@2: {bleu_score_2:.4f}, BLEU@3: {bleu_score_3:.4f}, BLEU@4: {bleu_score_4:.4f}, BLEU-SCORES: {bleu_score_total:.4f}")

        avg_epoch_loss = total_loss / len(train_loader)
        avg_bleu_1 = total_bleu_1 / len(train_loader)
        avg_bleu_2 = total_bleu_2 / len(train_loader)
        avg_bleu_3 = total_bleu_3 / len(train_loader)
        avg_bleu_4 = total_bleu_4 / len(train_loader)
        avg_bleu_scores = total_bleu_scores / len(train_loader)

        losses.append(avg_epoch_loss)
        bleu1_scores.append(avg_bleu_1)
        bleu2_scores.append(avg_bleu_2)
        bleu3_scores.append(avg_bleu_3)
        bleu4_scores.append(avg_bleu_4)
        bleu_scores.append(avg_bleu_scores)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Average Loss: {avg_epoch_loss:.4f}, "
              f"Average BLEU@1: {avg_bleu_1:.4f}, Average BLEU@2: {avg_bleu_2:.4f}, "
              f"Average BLEU@3: {avg_bleu_3:.4f}, Average BLEU@4: {avg_bleu_4:.4f}, "
              f"Average BLEU-Scores: {avg_bleu_scores:.4f}\n")

    return losses, bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores, bleu_scores


if __name__=="__main__":
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_DIR)
    model = VQAModel().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=int(len(train_loader) * 20)
    )

    ans_model = AnsEmbedding().to(config.DEVICE)
    train_model(model, ans_model, train_loader, tokenizer, criterion, optimizer, device=config.DEVICE, num_epochs=14, batch_size=config.BATCH_SIZE)