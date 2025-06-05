import networkx as nx
from ge import SDNE
from sklearn.cluster import KMeans
from sklearn.svm import SVR
import numpy as np
import random

def quota_samples(cluster, num_samples):
    """ 从一个簇中按配额采样 `num_samples` 个节点 """
    return random.sample(list(cluster), min(len(cluster), num_samples))


def SIR(G, seed_nodes, beta=0.042, gamma=0.01, iterations=100):
    """ SIR 扩散模型（更复杂版），根据种子节点、阈值以及其他参数返回节点的活力值 """

    # 初始化活力值
    vitality = {node: 0.0 for node in G.nodes()}
    active_nodes = set(seed_nodes)
    new_active_nodes = set(seed_nodes)

    for _ in range(iterations):
        next_active_nodes = set()
        for node in new_active_nodes:
            # 每个活动节点根据感染率传播感染
            for neighbor in G.neighbors(node):
                if neighbor not in active_nodes and random.random() < beta:
                    next_active_nodes.add(neighbor)
            # 恢复状态（活动节点变为非活动）
            if random.random() < gamma:
                active_nodes.remove(node)

        new_active_nodes = next_active_nodes
        active_nodes.update(new_active_nodes)

        # 更新节点活力值
        for node in active_nodes:
            vitality[node] += 1

    # Normalize vitality values
    max_vitality = max(vitality.values(), default=1)
    vitality = {node: v / max_vitality for node, v in vitality.items()}

    return vitality


# def SIR(seed_nodes, threshold):
#     """ SIR 扩散模型（简化版），根据种子节点和阈值返回节点的活力值 """
#     # 这里假设的 SIR 扩散模型，只是示例，实际可根据需求扩展。
#     vitality = {node: random.uniform(0, 1) for node in seed_nodes}
#     return vitality

if __name__ == "__main__":
    r = 0.01  # 采样比例
    k = 5  # 种子节点数
    b = 0.5  # 扩散阈值

    # Load graph
    G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

    # Train SDNE model
    model = SDNE(G, hidden_size=[256, 128])
    model.train(batch_size=3000, epochs=40, verbose=2)
    embeddings = model.get_embeddings()

    # 将嵌入转换为 NumPy 数组
    embedding_values = np.array(list(embeddings.values()))
    node_list = list(embeddings.keys())
    # Embedding = SDNE (G)

    # S_num = len(G.nodes()) * r
    S_num = int(len(G.nodes()) * r)

    # S_nodes = []
    S_nodes = []

    # C_list = K-means (embeddings, k)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(np.array(list(embeddings.values())))
    C_list = [np.where(kmeans.labels_ == i)[0] for i in range(k)]

    # For c in C_list: S_nodes.append(quota_samples(c, S_num))
    for c in C_list:
        S_nodes.extend(quota_samples(c, S_num))

    # Vitality = SIR(S_nodes, b)
    Vitality = SIR(S_nodes, b)

    # SVR_model = SVR(Embedding[S_nodes], Vitality)
    svr_model = SVR(kernel='linear')
    # X_train = np.array([embeddings[node] for node in S_nodes])
    X_train = np.array([embeddings[node_list[node]] for node in S_nodes])
    y_train = np.array([Vitality[node] for node in S_nodes])
    svr_model.fit(X_train, y_train)

    # 对所有节点进行活力值预测
    predicted_vitality = {}
    for node in G.nodes():
        if node in embeddings:
            vitality_value = svr_model.predict([embeddings[node]])
            predicted_vitality[node] = vitality_value[0]
    print("Vitality of nodes:",predicted_vitality)
    # 选择活力值最大的 K 个节点
    top_k_nodes = sorted(predicted_vitality, key=predicted_vitality.get, reverse=True)[:k]

    print("Top K nodes with highest predicted vitality:", top_k_nodes)

