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
    texts_inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, return_overflowing_tokens=True, max_length=max_length, stride=stride)
    if "overflow_to_sample_mapping" not in texts_inputs:
        texts_inputs['overflow_to_sample_mapping'] = [0]
    
    with torch.no_grad():
        anchor_embedding = model(**{k: v for k, v in anchor_inputs.items() if k != "overflow_to_sample_mapping"}).last_hidden_state
        texts_embeddings = model(**{k: v for k, v in texts_inputs.items() if k != "overflow_to_sample_mapping"}).last_hidden_state
        
        anchor_embedding = anchor_embedding[:, 0, :].mean(dim=0)

        mapping = texts_inputs['overflow_to_sample_mapping']
        cls_per_text = []
        cls_tokens = texts_embeddings[:, 0, :]
        current_cls = []
        current_idx = mapping[0]
        for i in range(len(mapping)):
            idx = mapping[i]
            if idx != current_idx:
                avg_cls = torch.stack(current_cls, dim=0).mean(dim=0)
                cls_per_text.append(avg_cls)
                current_cls = []
                current_idx = idx
            current_cls.append(cls_tokens[i])
        if current_cls:
            avg_cls = torch.stack(current_cls, dim=0).mean(dim=0)
            cls_per_text.append(avg_cls)
        texts_embeddings = torch.stack(cls_per_text, dim=0)
    
    # Compute cosine similarity between anchor and each text
    similarities = cosine_similarity(anchor_embedding.unsqueeze(0).numpy(), texts_embeddings.numpy())
    similarities = similarities[0]

    return similarities.flatten()

def rank(model, anchor, texts):
    sim_scores = compute_similarity(model, anchor, texts)
    ranks = rankdata(-sim_scores, method='ordinal')  

    return ranks

# # Example usage
model_name = "bert-base-uncased"
anchor_text = "This is the anchor sentence."
texts = [
    "This is the first text.",
    "I am.",
    "A completely different text."
]

similarities = compute_similarity(model_name, anchor_text, texts)
print(similarities)

ranks = rank(model_name, anchor_text, texts)
print(ranks)
