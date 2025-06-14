# ExVQA

This is the source code of the published paper CAEE_110439 ( ExVQA: a novel stacked attention networks with extended long short-term memory model for visual question answering) on the Computers and Electrical Engineering Journal.
[Link paper](https://www.sciencedirect.com/science/article/pii/S0045790625003829)

ExVQA is a Visual Question Answering (VQA) model that combines image and text embeddings with advanced neural network architectures, including Stacked Attention Networks (SANs) and xLSTM decoders, to generate answers to questions based on input images.


---

## System Requirements

- **Python**: 3.10
- **Dependencies**: Listed in `requirements.txt`
- **Hardware**: A GPU is recommended for training and inference.

---

## Environment Setup

1. **Clone the Repository**  
   Open your terminal and run:
   ```bash
   git clone https://github.com/Hovohoangduy/ExVQA.git
   cd ExVQA
   ```

2. **Install Dependencies**  
   Use `pip` to install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## Project Structure

- `config.py`: Configuration file for device, batch size, and model parameters.
- `data_processing.py`: Handles dataset loading and preprocessing.
- `features_extraction.py`: Defines image and question embedding models.
- `sans.py`: Implements Stacked Attention Networks (SANs) for image-text attention.
- `xlstm_decoder.py`: Contains the xLSTM decoder for sequence generation.
- `vqa_model.py`: Combines all components into the VQA model.
- `train.py`: Script for training the VQA model.
- `test.py`: Script for testing and inference.

---

## Training the Model

To train the ExVQA model, run the following command:
```bash
python train.py
```

### Training Parameters
- **Batch Size**: Configurable in `config.py` (default: 4).
- **Number of Epochs**: Configurable in `train.py` (default: 14).
- **Learning Rate**: Configurable in `train.py` (default: 0.0001).

---

## Testing the Model

To test the model on new images and questions, use the `test.py` script:
```bash
python test.py
```

### Example Usage
Uncomment the example code in `test.py` to test the model with sample images and questions.

---

## BLEU Score Evaluation

The training script calculates BLEU scores (BLEU@1, BLEU@2, BLEU@3, BLEU@4) to evaluate the model's performance. These scores are printed after each epoch.

---

## Dataset

The dataset should be structured with `train`, `validation`, and `test` splits. Update the `dataset` path in `data_processing.py` to point to your dataset.
