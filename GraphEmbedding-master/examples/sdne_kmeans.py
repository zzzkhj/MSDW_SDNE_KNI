# import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import networkx as nx
from ge.classify import read_node_label, Classifier
from ge import SDNE
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
from sklearn.metrics import pairwise_distances

from scipy.stats import pearsonr
#在网络划分后选取类簇中的核心节点
def select_seed_nodes(graph, embeddings, k):
    # Calculate degree centrality for each node
    degree_centrality = nx.degree_centrality(graph)
    print(degree_centrality)
    # Convert embeddings to a list
    emb_list = np.array([embeddings[node] for node in embeddings])

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=0).fit(emb_list)
    clusters = kmeans.predict(emb_list)

    seed_nodes = []
    for i in range(k):
        # Get nodes in the current cluster
        cluster_nodes = [node for node, label in zip(embeddings.keys(), clusters) if label == i]

        # Select the node with the highest degree centrality in the cluster
        seed_node = max(cluster_nodes, key=lambda node: degree_centrality[node])
        seed_nodes.append(seed_node)

    return seed_nodes

###一个簇中的节点如果与其他所有节点的相似度最大，那么它就是最重要的。基于向量表示的节点间相似度就是节点间的距离，距离越小，节点越相似。因此，在簇中，希望找到与其他节点距离最小的节点。
def select_seed_nodes_distance(graph, embeddings, k):
    # Convert embeddings to a list
    emb_list = np.array([embeddings[node] for node in embeddings])

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=0).fit(emb_list)
    clusters = kmeans.predict(emb_list)

    seed_nodes = []
    for i in range(k):
        # Get nodes in the current cluster
        cluster_nodes = [node for node, label in zip(embeddings.keys(), clusters) if label == i]

        # Calculate the distance sum for each node in the cluster
        min_distance_sum = float('inf')
        seed_node = None
        for node in cluster_nodes:
            distance_sum = sum(
                np.linalg.norm(embeddings[node] - embeddings[other_node]) for other_node in cluster_nodes if
                other_node != node)
            if distance_sum < min_distance_sum:
                min_distance_sum = distance_sum
                seed_node = node

        seed_nodes.append(seed_node)

    return seed_nodes


# def borda_count_ranking(scores):
#     # Borda Count assigns the highest rank to the best score
#     ranked_nodes = sorted(scores, key=scores.get, reverse=True)
#     borda_scores = {node: rank for rank, node in enumerate(ranked_nodes, start=1)}
#     return borda_scores
#
#
# def select_seed_nodes_borda_optimized(graph, embeddings, k):
#     # Calculate degree centrality for each node
#     degree_centrality = nx.degree_centrality(graph)
#
#     # Convert embeddings to a list
#     emb_list = np.array([embeddings[node] for node in embeddings])
#
#     # Perform K-means clustering
#     kmeans = KMeans(n_clusters=k, random_state=0).fit(emb_list)
#     clusters = kmeans.predict(emb_list)
#
#     seed_nodes = []
#     for i in range(k):
#         # Get nodes in the current cluster
#         cluster_nodes = [node for node, label in zip(embeddings.keys(), clusters) if label == i]
#
#         # 1. Rank nodes by degree centrality in the cluster
#         degree_scores = {node: degree_centrality[node] for node in cluster_nodes}
#         degree_borda_scores = borda_count_ranking(degree_scores)
#
#         # 2. Rank nodes by similarity (inverse of distance sum) in the cluster
#         cluster_emb_list = np.array([embeddings[node] for node in cluster_nodes])
#
#         # Vectorized computation of distance sums
#         distances = np.linalg.norm(cluster_emb_list[:, np.newaxis] - cluster_emb_list, axis=2)
#         distance_sums = np.sum(distances, axis=1)
#
#         # Convert distance sums to similarity scores (smaller distance sum means higher similarity)
#         similarity_scores = {node: -distance_sums[idx] for idx, node in enumerate(cluster_nodes)}
#         similarity_borda_scores = borda_count_ranking(similarity_scores)
#
#         # 3. Combine Borda Count from both rankings
#         total_borda_scores = {node: degree_borda_scores[node] + similarity_borda_scores[node]
#                               for node in cluster_nodes}
#
#         # Select the node with the highest Borda Count score
#         seed_node = max(total_borda_scores, key=total_borda_scores.get)
#         seed_nodes.append(seed_node)
#
#     return seed_nodes



def select_seed_nodes_kshell(graph, embeddings, k):
    # Step 3: Remove self-loops
    graph.remove_edges_from(nx.selfloop_edges(graph))

    # Calculate degree centrality for each node
    degree_centrality = nx.degree_centrality(graph)

    # Convert embeddings to a list
    emb_list = np.array([embeddings[node] for node in embeddings])

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=0).fit(emb_list)
    clusters = kmeans.predict(emb_list)

    # Group nodes by clusters
    cluster_nodes = {}
    for node, label in zip(embeddings.keys(), clusters):
        if label not in cluster_nodes:
            cluster_nodes[label] = []
        cluster_nodes[label].append(node)

    # Sort clusters by size in descending order
    sorted_clusters = sorted(cluster_nodes.items(), key=lambda item: len(item[1]), reverse=True)
    sorted_clusters = [list(item) for item in sorted_clusters]  # 将元组转换为列表

    # Step 4: Calculate K-shell values for all nodes
    k_shell_values = nx.core_number(graph)

    seed_nodes = []
    largest_cluster_size = len(sorted_clusters[0][1])

    # Initialize the number of nodes to be removed
    num_nodes_to_remove = 0

    # Step 4: Iteratively select nodes
    while True:
        seed_nodes = []
        for i, (cluster_label, nodes) in enumerate(sorted_clusters):
            cluster_size = len(nodes)

            # Check if the largest cluster's size is within K times of the current cluster's size
            if cluster_size * k >= largest_cluster_size or k == 1:
                # Select the node with the highest K-shell value in the cluster
                seed_node = max(nodes, key=lambda node: k_shell_values[node])
                seed_nodes.append(seed_node)

        # Check if the selected seed nodes meet the criteria
        if len(seed_nodes) == k:
            break

        # Step 5: Remove nodes from clusters that do not meet the criteria
        for i, (cluster_label, nodes) in enumerate(sorted_clusters):
            cluster_size = len(nodes)
            if cluster_size * k < largest_cluster_size:
                num_nodes_to_remove = int((largest_cluster_size - cluster_size * k) / k)
                sorted_clusters[i][1] = sorted_clusters[i][1][:-num_nodes_to_remove]

    return seed_nodes


def calculate_embedding_betweenness_centrality(embeddings):
    # Calculate the pairwise distance matrix of embeddings
    dist_matrix = pairwise_distances(embeddings)
    num_nodes = dist_matrix.shape[0]

    # Create a fully connected graph based on the distance matrix
    graph = nx.Graph()
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            graph.add_edge(i, j, weight=dist_matrix[i][j])

    # Calculate betweenness centrality based on shortest paths in embedding space
    return nx.betweenness_centrality(graph, weight='weight')


def calculate_embedding_closeness_centrality(embeddings):
    # Calculate the pairwise distance matrix of embeddings
    dist_matrix = pairwise_distances(embeddings)

    # Calculate closeness centrality based on the distance matrix in embedding space
    inv_dist_matrix = 1 / (dist_matrix + np.finfo(float).eps)  # Avoid division by zero
    closeness_centrality = inv_dist_matrix.sum(axis=1)
    return {i: c for i, c in enumerate(closeness_centrality)}


def select_seed_nodes_borda_count(graph, embeddings, k):
    # Calculate degree centrality in the original graph
    degree_centrality = nx.degree_centrality(graph)

    # Convert embeddings dictionary to a list of embedding vectors
    emb_list = np.array([embeddings[node] for node in embeddings])
    node_list = list(embeddings.keys())

    # Calculate betweenness centrality in embedding space
    betweenness_centrality = calculate_embedding_betweenness_centrality(emb_list)

    # Calculate closeness centrality in embedding space
    closeness_centrality = calculate_embedding_closeness_centrality(emb_list)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=0).fit(emb_list)
    clusters = kmeans.predict(emb_list)

    # Initialize dictionary to store Borda scores
    borda_scores = {node: 0 for node in graph.nodes}

    # Rank nodes based on degree centrality and assign Borda scores
    degree_rank = sorted(degree_centrality, key=degree_centrality.get, reverse=True)
    for rank, node in enumerate(degree_rank):
        borda_scores[node] += rank

    # Rank nodes based on betweenness centrality in embedding space and assign Borda scores
    betweenness_rank = sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)
    for rank, node_index in enumerate(betweenness_rank):
        borda_scores[node_list[node_index]] += rank

    # Rank nodes based on closeness centrality in embedding space and assign Borda scores
    closeness_rank = sorted(closeness_centrality, key=closeness_centrality.get, reverse=True)
    for rank, node_index in enumerate(closeness_rank):
        borda_scores[node_list[node_index]] += rank

    # Select seed nodes
    seed_nodes = []
    for i in range(k):
        # Get nodes in the current cluster
        cluster_nodes = [node for node, label in zip(node_list, clusters) if label == i]

        # Select the node with the highest Borda score in the cluster
        seed_node = max(cluster_nodes, key=lambda node: borda_scores[node])
        seed_nodes.append(seed_node)

    return seed_nodes


def borda_count_ranking(rankings):
    """合并多个排名使用 Borda Count 方法"""
    scores = {}
    for ranking in rankings:
        for i, node in enumerate(ranking):
            if node not in scores:
                scores[node] = 0
            scores[node] += i  # 累加每个节点的排名分数
    # 按分数排序，分数越小排名越靠前
    return sorted(scores.items(), key=lambda x: x[1])


def get_k_max_influence_nodes_borda(G, k, model):
    # Step 1: 获取节点的嵌入
    embeddings = model.get_embeddings()  # 获取所有节点的嵌入向量

    # Step 2: 计算每个节点的度中心性
    degree_centrality = nx.degree_centrality(G)

    # 将度中心性按降序排序，得到度中心性排名
    degree_ranking = sorted(degree_centrality, key=degree_centrality.get, reverse=True)

    # Step 3: 计算每个节点的皮尔逊相似度（基于嵌入向量）
    pearson_influence_dict = {}

    for i, node in enumerate(G.nodes, 1):
        node_influence = 0  # 初始化当前节点的综合影响为0
        for sim_node in G.nodes:
            if node != sim_node:  # 避免计算自己和自己的相似度
                pearson_corr, _ = pearsonr(embeddings[node], embeddings[sim_node])  # 使用节点嵌入
                node_influence += pearson_corr  # 累加皮尔逊相关系数
        node_influence /= (G.number_of_nodes() - 1)  # 计算平均值
        pearson_influence_dict[node] = node_influence


    # 将皮尔逊相似度按降序排序，得到皮尔逊排名
    pearson_ranking = sorted(pearson_influence_dict, key=pearson_influence_dict.get, reverse=True)

    # Step 4: 使用 Borda Count 方法合并排名
    combined_ranking = borda_count_ranking([degree_ranking, pearson_ranking])

    # 选取前 k 个节点作为最具影响力的节点
    top_k_nodes = np.array(combined_ranking)[:k, 0]

    return top_k_nodes


if __name__ == "__main__":
    # Load graph
    G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

    # Train SDNE model
    model = SDNE(G, hidden_size=[256, 128])
    model.train(batch_size=3000, epochs=40, verbose=2)
    embeddings = model.get_embeddings()

    # Select seed nodes using K-means
    k = 10
    seed_nodes = select_seed_nodes(G,embeddings, k)
    print("Selected seed nodes(use degree):", seed_nodes)
    # seed_nodes = get_k_max_influence_nodes_borda(G, k, model)
    # print("Selected seed nodes(use borde count):", seed_nodes)
    # seed_nodes = select_seed_nodes_borda_count(G, embeddings, k)
    # print("Selected seed nodes(use borde count):", seed_nodes)
    seed_nodes = select_seed_nodes_distance(G, embeddings, k)
    print("Selected seed nodes(use distance):", seed_nodes)
    # seed_nodes_kshell = select_seed_nodes_kshell(G,embeddings, k)
    # print("Selected seed nodes(use k-shell):", seed_nodes_kshell)
