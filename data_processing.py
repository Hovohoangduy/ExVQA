from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from config import config

dataset = "path/datasets"

transforms = transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor()
                                ])

class VQA_dataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]['image'].convert('RGB')
        question = self.data[idx]['question']
        answer = self.data[idx]['answer']
        if self.transform:
            image = self.transform(image)
        return image, question, answer
    
train_data = dataset['train']
val_data = dataset['validation']    
test_data = dataset['test']

train_dataset = VQA_dataset(train_data, transform=transforms)
val_dataset = VQA_dataset(val_data, transform=transforms)
test_dataset = VQA_dataset(test_data, transform=transforms)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

if __name__ == "__main__":
    random_indices = np.random.choice(len(train_dataset), 3)
    for idx in random_indices:
        idx = int(idx)  # Chuyển đổi chỉ số thành kiểu int
        image, question, answer = train_dataset[idx]
        image = image.permute(1, 2, 0).numpy()

        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title(f"Question: {question} \n Answer: {answer}", fontsize=16)
        plt.axis('off')
        plt.show()
    print(type(dataset['train']['question'][0]))