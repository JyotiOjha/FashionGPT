# -*- coding: utf-8 -*-
"""prompttotag.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RcEzQi8_GPa5VzfMBka2DaMoFH1_pFB3
"""



!pip install transformers

!pip install torch



import re
import nltk
import torch
nltk.download('stopwords')
from nltk.corpus import stopwords

from transformers import BertTokenizer, BertForMaskedLM

# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

# Preprocess prompt
def preprocess_prompt(prompt):
    stop_words = set(stopwords.words('english'))
    prompt = re.sub(r'[^\w\s]', ' ', prompt.lower())
    tokens = prompt.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

user_prompt = input("Please enter your prompt: ")
preprocessed_prompt = preprocess_prompt(user_prompt)

# Tokenize the preprocessed prompt
tokens = tokenizer.tokenize(preprocessed_prompt)

# Generate predictions using BERT's masked language model
indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
with torch.no_grad():
    predictions_logits = model(torch.tensor([indexed_tokens])).logits

# Get the predicted indices for the entire prompt
predicted_indices = torch.argmax(predictions_logits[0], dim=1).tolist()

# Convert the predicted indices back to tokens
predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_indices)

print("Predicted Keywords:", predicted_tokens)

