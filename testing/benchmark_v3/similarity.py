import torch
from torch.nn import CosineSimilarity
from abc import ABC, abstractmethod
device = torch.device("cuda")

class EmbSummarizer(ABC):
    @abstractmethod
    def summarize(self, texts: list):
        pass

class BERTPooler(EmbSummarizer):
    def __init__(self, model, tokenizer, **kwargs):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        req_kwargs = {
            "max_length": 512,
            "stride": 10
        }
        self.tokenizer_args = {**req_kwargs, **kwargs}

    def embed(self, inputs):
        input_batch = {k: v.detach().clone().to(device) for k, v in inputs.items() if k in ['input_ids', 'attention_mask', 'token_type_ids']}
        with torch.no_grad():
            emb = self.model(**input_batch)
        return emb

    def tokenize(self, texts: list):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(texts, return_tensors='pt', truncation=True, return_overflowing_tokens=True, padding=True, **self.tokenizer_args)
        if "overflow_to_sample_mapping" not in inputs:
            inputs['overflow_to_sample_mapping'] = [0]
        return inputs

class BERTCLSFirstPooler(BERTPooler):
    """
    Creates embedding of texts using last cls
    """
    def tokenize(self, texts: list):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(texts, return_tensors='pt', truncation=True, padding=True, **self.tokenizer_args)
        return inputs

    def summarize(self, texts: list):
        inputs = self.tokenize(texts)
        emb = self.embed(inputs)
        return emb.last_hidden_state[:, 0]

class BERTCLSMeanPooler(BERTPooler):
    """
    Creates embedding of texts using mean of cls across chunks
    """
    def summarize(self, texts: list):
        inputs = self.tokenize(texts)
        emb = self.embed(inputs)
        emb_cls = emb.last_hidden_state[:, 0]
        mapping = inputs['overflow_to_sample_mapping']
        mean_cls_per_text = []
        current_cls = []
        current_id = mapping[0].item()
        for i in range(mapping.shape[0]):
            idx = mapping[i].item()
            if idx != current_id:
                mean_cls = torch.stack(current_cls, dim=0).mean(dim=0)
                mean_cls_per_text.append(mean_cls)
                current_cls = []
                current_id = idx
            current_cls.append(emb_cls[i])
        if current_cls:
            mean_cls = torch.stack(current_cls, dim=0).mean(dim=0)
            mean_cls_per_text.append(mean_cls)
        
        return torch.stack(mean_cls_per_text, dim=0)

class SimScorer(ABC):
    def __init__(self, emb_summarizer: EmbSummarizer = BERTCLSMeanPooler):
        self.emb_summarizer = emb_summarizer
    
    def preprocess(self, texts: list):
        return texts
    
    @abstractmethod
    def calc_sim(self, texts: list):
        pass

    @abstractmethod
    def rank(self, texts: list):
        pass

class CosineSimScorer(SimScorer):
    def calc_sim(self, batch1, batch2):
        emb_cls1 = self.emb_summarizer.summarize(batch1).cpu()
        emb_cls2 = self.emb_summarizer.summarize(batch2).cpu()
        cos_sim = CosineSimilarity()
        similarity = cos_sim(emb_cls1, emb_cls2)
        return similarity
    
    def rank(self, anchor, texts: list):
        sim_scores = self.calc_sim([anchor] * len(texts), texts)
        ranks = torch.argsort(-sim_scores, dim=0)
        return ranks
    
# from transformers import AutoTokenizer, AutoModel
# model_name = "bert-base-uncased"
# model = AutoModel.from_pretrained(model_name).to(device)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# texts = [
#     "This is the first text." * 100,
#     "I am." * 100,
#     "A completely different text." * 100
# ]

# kwargs = {
#     "max_length": 512,
#     "stride": 10
# }

# summarizers = {"Mean Pooling": BERTCLSMeanPooler,
#                "First Chunk": BERTCLSFirstPooler}
# for s_name in summarizers:
#     s_c = summarizers[s_name]
#     summarizer = s_c(model, tokenizer, **kwargs)
#     sim_scorer = CosineSimScorer(summarizer)
#     print(s_name, sim_scorer.calc_sim(["I am." * 100] * 3, texts))
#     print(sim_scorer.rank("I am." * 100, texts))
#     print()
