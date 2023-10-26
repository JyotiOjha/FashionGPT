import re
import nltk
import torch
import pandas as pd
nltk.download('stopwords')
from nltk.corpus import stopwords

from transformers import BertTokenizer, BertForMaskedLM

# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

# Load the dataset
df = pd.read_csv('/content/Womens_Clothing_E-Commerce_Reviews[1].csv')

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

# Suggested tags
tags = predicted_tokens

# Filter products based on extracted tags
# Filter products based on extracted tags
suggested_products = df[df['Review Text'].apply(lambda x: isinstance(x, str) and any(tag in x.lower() for tag in tags))]

# Print the suggested products
print("Suggested Products:")
for idx, product in suggested_products.iterrows():
    print(f"- Product ID: {product['Clothing ID']}")
    print(f"  Title: {product['Title']}")
    print(f"  Description: {product['Review Text']}")
    print()

