from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import rankdata

def chunk(tokens, max_length=512, stride=100):
    chunks = [tokens[i:i+max_length] for i in range(0, tokens.shape[0], max_length - stride)]



def compute_similarity(model_name, anchor, texts):
    # Load the tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    max_length = model.config.max_position_embeddings
    stride = 100

    # Tokenize the anchor and texts
    anchor_inputs = tokenizer(anchor, return_tensors='pt', padding=True, truncation=True, return_overflowing_tokens=True, max_length=max_length, stride=stride)
    texts_inputs = []
    for text in texts:
        texts_inputs.append(tokenizer(texts, return_tensors='pt', padding=True, truncation=True, return_overflowing_tokens=True, max_length=max_length, stride=stride))
    print(tokenizer(texts, return_tensors='pt', padding=True, truncation=True, return_overflowing_tokens=True, max_length=max_length, stride=stride))

    with torch.no_grad():
        anchor_embedding = model(**anchor_inputs).last_hidden_state[:, 0, :].mean(dim=0)
        texts_embeddings = []
        for text_input in texts_inputs:
            texts_embeddings.append(model(**text_input).last_hidden_state[:, 0, :].mean(dim=0))
    
    print(anchor_embedding, texts_embeddings)
    # Compute cosine similarity between anchor and each text
    similarities = cosine_similarity(anchor_embedding.numpy(), texts_embeddings.numpy())
    
    return similarities.flatten()

def rank(model, anchor, texts):
    sim_scores = compute_similarity(model, anchor, texts)
    ranks = rankdata(-sim_scores, method='ordinal')  

    return ranks

# # Example usage
model_name = "bert-base-uncased"
anchor_text = "This is the anchor sentence." * 1000
texts = [
    "This is the first text." * 1000,
    "I am." * 1000,
    "A completely different text." * 1000
]

similarities = compute_similarity(model_name, anchor_text, texts)
print(similarities)

# ranks = rank(model_name, anchor_text, texts)
# print(ranks)
