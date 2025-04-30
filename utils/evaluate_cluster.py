import time
from sentence_transformers import util

import plotly.graph_objects as go
import numpy as np
import matplotlib.cm as cm
from plotly.subplots import make_subplots
from sklearn import metrics

from sklearn.manifold import TSNE
    

def cluster_corpus(corpus_sentences, model, min_community_size=25, threshold=0.75, log=True):
    corpus_embeddings = model.encode(corpus_sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True)

    if log:
        print("Start clustering")
    start_time = time.time()

    # Two parameters to tune:
    # min_cluster_size: Only consider cluster that have at least 25 elements
    # threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
    clusters = util.community_detection(corpus_embeddings, min_community_size=min_community_size, threshold=threshold)

    if log:
        print(f"Clustering done after {time.time() - start_time:.2f} sec")
    return clusters

def visualise_clusters(clusters_gt, clusters_pred, corpus_embeddings_2d):
    colors_pred = cm.tab20(np.linspace(0, 1, len(clusters_pred)))
    colors_gt = cm.tab20(np.linspace(0, 1, len(clusters_gt)))

    def rgba_to_hex(rgba):
        r, g, b, _ = (int(255 * x) for x in rgba)
        return f'#{r:02x}{g:02x}{b:02x}'

    colors_pred = [rgba_to_hex(c) for c in colors_pred]
    colors_gt = [rgba_to_hex(c) for c in colors_gt]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Clusters of 2D Embeddings gt", "Clusters of 2D Embeddings pred"))

    for i, cluster in enumerate(clusters_gt):
        points = corpus_embeddings_2d[cluster]
        fig.add_trace(
            go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                mode='markers',
                marker=dict(color=colors_gt[i], size=6),
                name=f"GT Group {i}",
                legendgroup="gt",
                showlegend=False
            ),
            row=1, col=1
        )

    for i, cluster in enumerate(clusters_pred):
        points = corpus_embeddings_2d[cluster]
        fig.add_trace(
            go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                mode='markers',
                marker=dict(color=colors_pred[i], size=6),
                name=f"Pred Group {i}",
                legendgroup="pred",
                showlegend=False
            ),
            row=1, col=2
        )

    fig.update_layout(
        height=600,
        width=1000,
        title_text="Cluster Comparison: Ground Truth vs Prediction",
        showlegend=True
    )

    return fig

def labels_to_metrics(labels_pred, labels_gt):
    metric_d = {}
    metric_d["random score"] = metrics.rand_score(labels_pred, labels_gt)
    metric_d['adjusted random score'] = metrics.adjusted_rand_score(labels_pred, labels_gt)

    return metric_d

def clustering_eval(df, model, min_community_size=25, threshold=0.75, log=True):
    corpus_sentences = df['text'].tolist()
    clusters = cluster_corpus(corpus_sentences=corpus_sentences, model=model, min_community_size=min_community_size, threshold=threshold, log=log)

    clusters_gt = df.reset_index().groupby("label")[['index']].agg(list)['index'].tolist()
    clusters_pred = clusters

    corpus_embeddings = model.encode(corpus_sentences)
    corpus_embeddings_2d = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(corpus_embeddings)

    fig = visualise_clusters(clusters_gt=clusters_gt, clusters_pred=clusters_pred, corpus_embeddings_2d=corpus_embeddings_2d)

    labels_gt = df['label'].tolist()
    labels_pred = [-1] * len(labels_gt)
    for id, cluster in enumerate(clusters):
        for i in cluster:
            labels_pred[i] = id

    metric_d = labels_to_metrics(labels_pred, labels_gt)

    return clusters, fig, metric_d