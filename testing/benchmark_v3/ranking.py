from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import rankdata

def compute_similarity(model, anchor, texts):
    # Load the tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModel.from_pretrained(model)
    
    # Tokenize the anchor and texts
    anchor_inputs = tokenizer(anchor, return_tensors='pt', padding=True, truncation=True, max_length=512)
    texts_inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    # Get the embeddings for the anchor and the texts
    with torch.no_grad():
        anchor_embedding = model(**anchor_inputs).last_hidden_state.mean(dim=1)
        texts_embeddings = model(**texts_inputs).last_hidden_state.mean(dim=1)
    
    # Compute cosine similarity between anchor and each text
    similarities = cosine_similarity(anchor_embedding.numpy(), texts_embeddings.numpy())
    
    return similarities.flatten()

def rank(model, anchor, texts):
    sim_scores = compute_similarity(model, anchor, texts)
    ranks = rankdata(-sim_scores, method='ordinal')  

    return ranks

# Example usage
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
