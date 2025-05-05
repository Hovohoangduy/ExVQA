import torch

class config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 4
    IMG_DIR = 'google/siglip-base-patch16-224'
    TEXT_DIR = 'EleutherAI/gpt-neo-125m'
    MAX_LEN = 64
    d_model = 768