
import pandas as pd
import os
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda()

def check_similarity(queries, docs):
    query_prompt_name = "s2p_query"

    query_embeddings = model.encode(queries, prompt_name=query_prompt_name)
    doc_embeddings = model.encode(docs)
    print(query_embeddings.shape, doc_embeddings.shape)

    similarities = model.similarity(query_embeddings, doc_embeddings)
    return similarities


problems = []
for i in range(2002, 2025):
    entry = {}
    entry['year'] = i
    for j in range(1, 7):
        problem_path = os.path.abspath(f'./IMO-{i}-notes/Problem_{j}.tex')
        with open(problem_path, 'r') as f:
            entry[f'problem_{j}'] = f.read()

        solution_path = os.path.abspath(f"./IMO-{i}-notes/{i}_P{j}.tex")
        with open(solution_path, 'r') as f:
            entry[f'solution_{j}'] = f.read()
    problems.append(entry)

df = pd.DataFrame(problems)