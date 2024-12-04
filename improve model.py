import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    AutoTokenizer,
    EarlyStoppingCallback
)
import torch
import torch.nn.functional as F

# Sample data preparation (expanded dataset)
data = {
    'text': [
        "Information about your services.",
        "I would like to schedule an appointment.",
        "I wanted to know you business hours",
        "What services do you offer?",
        "hello Can i know about your services?",
        "Hi, What are your hours of operation?",
        "I would like to know more about your services.",
        "hi Can I book appointment?",
        "How much does AI app service cost?",
        "What time does your work hours start?",
        "Can I have contact number?",
        "I'd like to book for 20th dec at 2pm",
        "Can I book for 19 January at maybe 10am?",
        "Tell me about your company background.",
        "Do you provide consultation services?",
        "I need to reschedule my booking.",
        "What are your pricing details?",
        "Can we set up a meeting?",
        "Explain your service offerings."
    ],
    'label': [0, 1, 2, 0, 0, 2, 0, 1, 0, 2, 2, 1, 1, 0, 0, 1, 0, 1, 0]  
}
df = pd.DataFrame(data)

# Stratified split to maintain label distribution
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Advanced tokenization with more preprocessing
model_name = "distilbert-base-uncased"  # More efficient than BERT
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Custom preprocessing to clean and normalize text
def preprocess_text(text):
    """Basic text preprocessing"""
    return text.lower().strip()

train_df['text'] = train_df['text'].apply(preprocess_text)
val_df['text'] = val_df['text'].apply(preprocess_text)

# Tokenization with additional parameters
def tokenize_function(examples):
    return tokenizer(
        examples, 
        padding='max_length', 
        max_length=64,  # Adjust based on your typical input length
        truncation=True, 
        return_tensors='pt'
    )

train_encodings = tokenize_function(train_df['text'].tolist())
val_encodings = tokenize_function(val_df['text'].tolist())

# Custom Dataset with more robust implementation
class ImprovedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ImprovedDataset(train_encodings, train_df['label'].tolist())
val_dataset = ImprovedDataset(val_encodings, val_df['label'].tolist())

# Custom compute metrics function for more detailed evaluation
def compute_metrics(pred):
    """Compute detailed classification metrics"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Classification report
    report = classification_report(labels, preds, output_dict=True)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(labels, preds)
    
    # Return both classification metrics and confusion matrix
    return {
        'accuracy': report['accuracy'],
        'macro_precision': report['macro avg']['precision'],
        'macro_recall': report['macro avg']['recall'],
        'macro_f1': report['macro avg']['f1-score'],
        'confusion_matrix': conf_matrix.tolist()
    }

# Advanced training arguments with more optimization techniques
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,  # Increased epochs with early stopping
    per_device_train_batch_size=16,  # Potentially larger batch size
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,  # Load best model after training
    metric_for_best_model='eval_macro_f1',  # Use macro F1 for model selection
    learning_rate=5e-5,  # Slightly adjusted learning rate
    fp16=True,  # Mixed precision training
)

# Load model with label weights to handle class imbalance
class_weights = torch.tensor([1.0, 1.5, 1.2])  # Adjust based on your class distribution
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=3,
    # Optional: pass class weights if needed
    # problem_type="single_label_classification"
)

# Initialize Trainer with early stopping and advanced configurations
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Stop if no improvement
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# Save the trained model and tokenizer
trainer.save_model('./improved_model')
tokenizer.save_pretrained('./improved_model')