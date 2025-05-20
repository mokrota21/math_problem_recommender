# Probably not going to use it anymroe

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import rankdata

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.stats import rankdata
import numpy as np


def compute_nli(model_name, query, text):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # run through model pre-trained on MNLI
    x = tokenizer.encode(text, query, return_tensors='pt',
                        truncation_strategy='only_first')
    logits = model(x)[0]

    # we throw away "neutral" (dim 1) and take the probability of
    # "entailment" (2) as the probability of the label being true 
    entail_contradiction_logits = logits[:,[0,2]]
    probs = entail_contradiction_logits.softmax(dim=1)
    prob_label_is_true = probs[:,1]
    return prob_label_is_true.detach()

def compute_max_pooling_nli(model_name, query, text, max_length=124, stride=20):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length - stride)]
    inputs = tokenizer(chunks, [query] * len(chunks), return_tensors='pt', max_length=max_length, stride=stride, return_overflowing_tokens=True, padding=True)
    # print(len(chunks))
    # print(inputs['input_ids'].shape)

    max_prob = 0.0
    for i in range(inputs["input_ids"].size(0)):
        input_batch = {k: v[i].unsqueeze(0) for k, v in inputs.items() if k in ['input_ids', 'attention_mask', 'token_type_ids']}
        with torch.no_grad():
            logits = model(**input_batch).logits

        # we throw away "neutral" (dim 1) and take the probability of
        # "entailment" (2) as the probability of the label being true 
        entail_contradiction_logits = logits[:,[0,2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        prob_label_is_true = probs[:,1].item()
        max_prob = max(prob_label_is_true, max_prob)
    return max_prob

def compute_similarity_nli(model_name, query, texts, chunk=True, max_length=124, stride=20):
    compute = compute_max_pooling_nli if chunk else compute_nli
    entail_probs = np.array([compute(model_name, query, text, max_length, stride) for text in texts])
    return entail_probs

def compute_similarity_chunking_cls(model_name, anchor, texts):
    # Load the tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    max_length = model.config.max_position_embeddings
    stride = 20

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

def rank(kwargs, compute_similarity: callable):
    sim_scores = compute_similarity(**kwargs)
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

similarities = compute_similarity_chunking_cls(model_name, anchor_text, texts)
print(similarities)

# ranks = rank(model_name, anchor_text, texts)
# print(ranks)
