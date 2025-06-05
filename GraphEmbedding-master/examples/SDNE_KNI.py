import json
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.svm import SVR
from ge import SDNE
import time
import random
import pandas as pd
import sys
import logging
import random
from scipy.stats import kendalltau

def IC2(g, S, mc):

    spread = []
    for j in range(mc):
        new_active, A = S[:], S[:]
        while new_active:
            new_ones = []
            for node in new_active:
                for s in g.neighbors(node):
                    if s not in S:
                        # 第一种方法：获得邻居结点s的度分之一
                        m = 1 / g.degree(s)
                        if np.random.uniform(0, 1) < m:
                            new_ones.append(s)

            new_active = list(set(new_ones) - set(A))
            A += new_active

        spread.append(len(A))
    return np.mean(spread)


def SIR_model(G, initial_infected,beta=0.2, gamma=0.01, num_simulations=1000):
    if initial_infected is None:
        initial_infected = random.choice(list(G.nodes()))

    # initial_infected = set(initial_infected)  # 假设 initial_infected 是一个可以转换为整数的字符串
    # 选中当前节点作为初始感染节点
    # for i in initial_infected:
    total_spread = 0  # 累积传播能力

    for _ in range(num_simulations):
        S = set(G.nodes())  # Susceptible
        # I = {initial_infected}  # Infected
        I = set(initial_infected)
        R = set()  # Recovered
        for node in initial_infected:
            if node in S:
                S.remove(node)
        # S.remove(initial_infected)
        total_infected = set(I)  # 用于记录所有被感染的节点
        while (True):
            new_infected = set()
            new_recovered = set()
            # 遍历当前感染节点
            for node in I:
                # 感染邻居
                neighbors = G.neighbors(node)
                for neighbor in neighbors:
                    if neighbor in S and random.random() < beta:
                        new_infected.add(neighbor)

                # 节点可能康复
                if random.random() < gamma:
                    new_recovered.add(node)
            # 更新状态
            S -= new_infected
            I = (I - new_recovered) | new_infected
            R |= new_recovered
            # 记录总感染节点
            total_infected |= new_infected
            # 如果没有感染者，结束传播
            if len(I) == 0:
                break
        total_spread += len(total_infected)

        # 记录当前模拟中的传播能力（感染到的节点数）
    return total_spread / num_simulations

# def SIR_model(G, initial_infected, beta=0, gamma=0.01, iterations=50):
#     if initial_infected is None:
#         initial_infected = random.choice(list(G.nodes()))
#
#     S = set(G.nodes())  # Susceptible
#     I = {initial_infected}  # Infected
#     R = set()  # Recovered
#     S.remove(initial_infected)
#
#     for _ in range(iterations):
#         new_infected = set()
#         new_recovered = set()
#
#         # Try to infect neighbors of infected nodes
#         for node in I:
#             neighbors = G.neighbors(node)
#             for neighbor in neighbors:
#                 if neighbor in S and random.random() < beta:
#                     new_infected.add(neighbor)
#
#             # Process recovery of currently infected nodes
#             if random.random() < gamma:
#                 new_recovered.add(node)
#
#         S -= new_infected  # Remove newly infected from susceptible set
#         I = (I - new_recovered) | new_infected  # Update infected set
#         R |= new_recovered  # Add newly recovered to recovered set
#
#         # If no one is infected, stop the simulation
#         if len(I) == 0:
#             break
#
#     # Calculate the total number of infected (including recovered) nodes
#     total_infected = len(R)  # All recovered were once infected
#
#     # Return the fraction of nodes that were infected
#     # return total_infected / G.number_of_nodes()
#     return total_infected


# Vitality function using SIR model
def df(x_sample_indices, G, node_order):
    vitality = []
    for idx in x_sample_indices:
        # 检查索引是否在 node_order 范围内
        if idx < 0 or idx >= len(node_order):
            print(f"Warning: Index {idx} out of range for node_order.")
            continue
        node = node_order[idx]
        if node not in G:
            print(f"Warning: Node {node} not found in the graph.")
            continue  # 跳过不在图中的节点
        # vitality.append(SIR_model(G, initial_infected=node,beta=0.042, gamma=0.01,  iterations=50))
        vitality.append(SIR_model_spreadability[node])
        # vitality.append(SIR_model_spreadability(G, initial_infected=node, beta=0.037, gamma=0.01, num_simulations=1))
    return np.array(vitality)


#  score
def score(G, svr_model, X, node_order, alpha=0.25):
    EML = {}
    # dc = nx.degree_centrality(G)
    # degree_centrality_dict = {node: centrality for node, centrality in dc.items()}
    for idx, node in enumerate(node_order):
        svr_pred = svr_model.predict(X[idx].reshape(1, -1))[0]
        # print(idx,node)
        # print("当前节点的预测值是：",svr_pred)
        neighbor_sum = 0
        for neighbor in G.neighbors(node):
            if neighbor in node_order:
                neighbor_idx = node_order.index(neighbor)
                neighbor_pred = svr_model.predict(X[neighbor_idx].reshape(1, -1))[0]
                # neighbor_sum += neighbor_pred
                neighbor_degree = G.degree(neighbor)  # 获取邻居节点的度
                neighbor_sum += neighbor_degree * neighbor_pred  # 邻居度 * 邻居活力
                # h_index = h_index_centrality(G).get(neighbor)
                # neighbor_sum += h_index * neighbor_pred
        EML[node] = svr_pred + alpha * neighbor_sum
        # EML[node] = svr_pred
    # print("所有节点的最终得分是：", EML)

    sorted_nodes = sorted(EML, key=EML.get, reverse=True)
    # print(sorted_nodes)
    return sorted_nodes, EML


# SVR Prediction
def predict_vitality(svr_model, X, node_order):
    EML = {}
    for idx, node in enumerate(node_order):
        svr_pred = svr_model.predict(X[idx].reshape(1, -1))[0]

        EML[node] = svr_pred
    # sorted_nodes = sorted(EML, key=EML.get, reverse=True)
    return EML


# Gap Statistic（间隙统计量）方法，用于评估聚类分析中最佳聚类数量 k

def elbow_method(X, max_k=10):
    inertia_values = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X)
        inertia_values.append(kmeans.inertia_)
    # 找到肘部位置 (例如简单版，寻找Inertia下降最快的点)
    k_opt = np.argmin(np.diff(inertia_values)) + 1
    return k_opt


def gap_statistic(X, max_k=10, n_refs=10):
    """
    使用 Gap Statistic 方法评估最佳聚类数量 k.

    Parameters:
    - X: np.array, 形状为 (num_samples, num_features) 的嵌入向量
    - max_k: int, 评估的最大聚类数
    - n_refs: int, 随机参考数据集的数量

    Returns:
    - gaps: 各个聚类数量 k 对应的 Gap 值
    - optimal_k: 最佳聚类数量 k
    """
    # 存储 gap 值
    gaps = np.zeros(max_k - 1)
    # 存储每个 k 对应的实际数据的 inertia（簇内平方和）
    inertias = np.zeros(max_k - 1)
    # 存储随机数据的 inertia（簇内平方和）
    ref_inertias = np.zeros((n_refs, max_k - 1))
    # 存储标准误差
    s_k = np.zeros(max_k - 1)

    # 生成随机数据的边界
    shape_min = np.min(X, axis=0)
    shape_max = np.max(X, axis=0)

    for k in range(1, max_k):
        # 进行 k-means 聚类并计算实际数据的 inertia
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        inertias[k - 1] = kmeans.inertia_

        # 生成 n_refs 个随机参考数据集，并计算它们的 inertia
        for i in range(n_refs):
            random_data = np.random.uniform(shape_min, shape_max, X.shape)
            kmeans_random = KMeans(n_clusters=k)
            kmeans_random.fit(random_data)
            ref_inertias[i, k - 1] = kmeans_random.inertia_

        # 计算 gap 值
        gaps[k - 1] = np.log(np.mean(ref_inertias[:, k - 1])) - np.log(inertias[k - 1])

        # 计算标准误差
        s_k[k - 1] = np.sqrt(
            np.mean((np.log(ref_inertias[:, k - 1]) - np.log(np.mean(ref_inertias[:, k - 1]))) ** 2)) * np.sqrt(
            1 + 1 / n_refs)

    # 选择满足 Gap(k) >= Gap(k+1) - s_{k+1} 的最小 k
    for k in range(1, max_k - 1):
        if gaps[k - 1] >= gaps[k] - s_k[k]:
            optimal_k = k
            break
    else:
        optimal_k = max_k - 1  # 如果没有符合条件的k，则选择 max_k-1

    return optimal_k, gaps


def combined_optimal_k(X):
    k_elbow = elbow_method(X)
    k_gap, _ = gap_statistic(X)
    # 这里的 `opt` 可以是任何你定义的策略，比如取平均值
    optimal_k = (k_elbow + k_gap) // 2
    print("k_gap,k_elbow分别是：")
    print(k_gap, k_elbow)
    return optimal_k


def convert_to_N_by_1(array):
    # 获取原始数组的元素总数
    num_elements = array.size

    # 计算转化后数组的行数N
    N = num_elements

    # 利用reshape方法将数组转化为N×1形式
    # 注意：-1在reshape中用作占位符，表示该维度的大小由其他维度自动推断
    # 但在此处我们已明确知道N，因此直接指定即可
    reshaped_array = array.reshape(N, 1)

    return reshaped_array


def h_index_centrality(G):
    h_index = {}
    for node in G.nodes():
        # 获取邻居节点的度数
        neighbor_degrees = [G.degree(neighbor) for neighbor in G.neighbors(node)]
        # 对邻居节点的度数进行降序排序
        neighbor_degrees.sort(reverse=True)

        # 计算H-指数
        h = 0
        for i, degree in enumerate(neighbor_degrees):
            if degree >= i + 1:
                h = i + 1
            else:
                break
        h_index[node] = h
    return h_index


def jaccard_coefficient(nodes1, nodes2):
    set1 = set(nodes1)
    set2 = set(nodes2)

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection / union if union != 0 else 0
if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    # 创建一个handler
    handler = logging.StreamHandler()
    logger.setLevel(logging.DEBUG)  # 设置日志级别为DEBUG，这样所有级别的日志都会被处理
    # 创建一个格式器，并添加到handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # 给logger添加handler
    logger.addHandler(handler)

    network_name_dict = {'1': 'cora', '2': 'bio-CE-LC', '3': 'citeseer', '4': 'socfb-Reed98',
                         '5': 'ia-fb-messages', '6': 'bio-CE-GT', '7': 'ca-GrQc', '8': 'power-US-Grid','9':'soc-hamster'}
    network_name_key = input(
        '请输入选择使用的网络编号:1: cora, 2: bio-CE-LC, 3: citeseer, 4: socfb-Reed98,5: ia-fb-messages, 6: bio-CE-GT,7:ca-GrQc,8:power-US-Grid,9:soc-hamster')
    if network_name_key not in network_name_dict:
        logger.error('请输入网络编号')
        sys.exit(0)
    network_name = network_name_dict[network_name_key]

    if network_name == 'cora':
        data = pd.read_csv('data/cora.edges', sep=',', header=None)
        data.columns = ['source', 'target', 'flag']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
        logger.info(f'使用{network_name}数据集，图有{G.number_of_nodes()}个节点，{G.number_of_edges()}条边。')
        beta = 0.101
    elif network_name == 'bio-CE-LC':
        data = pd.read_csv('data/bio-CE-LC.edges', sep=' ', header=None)
        data.columns = ['source', 'target', 'flag']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
        logger.info(f'使用{network_name}数据集，图有{G.number_of_nodes()}个节点，{G.number_of_edges()}条边。')
        beta = 0.082
    elif network_name == 'citeseer':
        data = pd.read_csv('data/citeseer.txt', sep=',', header=None)
        data.columns = ['source', 'target', 'flag']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
        logger.info(f'使用{network_name}数据集，图有{G.number_of_nodes()}个节点，{G.number_of_edges()}条边。')
        beta = 0.3
    elif network_name == 'socfb-Reed98':
        data = pd.read_csv('data/socfb-Reed98.mtx', sep=' ', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
        logger.info(f'使用{network_name}数据集，图有{G.number_of_nodes()}个节点，{G.number_of_edges()}条边。')
        beta = 0.016
    elif network_name == 'ia-fb-messages':
        data = pd.read_csv('data/ia-fb-messages.mtx', sep=' ', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
        logger.info(f'使用{network_name}数据集，图有{G.number_of_nodes()}个节点，{G.number_of_edges()}条边。')
        beta = 0.04
    elif network_name == 'bio-CE-GT':
        data = pd.read_csv('data/bio-CE-GT.edges', sep=' ', header=None)
        data.columns = ['source', 'target', 'flag']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
        logger.info(f'使用{network_name}数据集，图有{G.number_of_nodes()}个节点，{G.number_of_edges()}条边。')
        beta = 0.037
    elif network_name == 'ca-GrQc':
        data = pd.read_csv('data/ca-GrQc.mtx', sep=' ', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
        logger.info(f'使用{network_name}数据集，图有{G.number_of_nodes()}个节点，{G.number_of_edges()}条边。')
        beta = 0.05
    elif network_name == 'power-US-Grid':
        data = pd.read_csv('data/power-US-Grid.mtx', sep=' ', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
        logger.info(f'使用{network_name}数据集，图有{G.number_of_nodes()}个节点，{G.number_of_edges()}条边。')
        beta = 0.3
    elif network_name == 'soc-hamster':
        data = pd.read_csv('data/soc-hamsterster.edges', sep=' ', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
        logger.info(f'使用{network_name}数据集，图有{G.number_of_nodes()}个节点，{G.number_of_edges()}条边。')
        results_filename = '{soc-hamster}_SIR_spreadability_results.json'
        beta = 0.03
    # print(f'{"{"}{network_name}{"}"}_SIR_spreadability_results.json')
    with open(f'{"{"}{network_name}{"}"}_SIR_spreadability_results.json', 'r') as f:
        # print(f.read())
        SIR_model_spreadability = json.load(f)

    start_time = time.time()

    # Step 1: Load graph and compute embeddings
    # G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt', create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    model = SDNE(G, hidden_size=[256, 128])
    model.train(batch_size=3000, epochs=100, verbose=2)
    embeddings = model.get_embeddings()
    # print("embeddings:",embeddings)
    node_order = list(embeddings.keys())
    # print("node_order是：", node_order)
    # print("node_order个数：", len(node_order))
    X = np.array([embeddings[node] for node in node_order])  # 使从X[0]开始的
    # print(X.shape)
    # print(X)
    # print(X[1])
    # 将嵌入向量列表转换为NumPy数组，这个数组的形状将是 (num_nodes, 128)，其中 num_nodes 是图中节点的数量，每个节点有128维的嵌入向量

    # Step 2: Compute the optimal number of clusters using gap statistic
    # optimal_k,_ = gap_statistic(X)
    # optimal_k = combined_optimal_k(X)
    optimal_k=5
    # optimal_k = elbow_method(X, max_k=10)
    # print(f"Optimal number of clusters (k): {optimal_k}")

    # Step 3: Cluster nodes and sample |s|/k nodes from each cluster
    kmeans = KMeans(n_clusters=optimal_k, random_state=0)
    clusters = kmeans.fit_predict(X)
    # print("clusters:",clusters)
    rate=0.01
    s = G.number_of_nodes() * rate
    sampled_x, sampled_y, sampled_nodes = [], [], []
    for i in range(optimal_k):
        cluster_indices = np.where(clusters == i)[0]  # cluster_indices就包含了所有被分配到第i个聚类的节点的索引。
        sample_size = int(s // optimal_k)

        # sample_size = min(len(cluster_indices), sample_size)
        sample_size = max(1, int(s // max(1, optimal_k)))

        # sample_indices = np.random.choice(cluster_indices, size=sample_size, replace=False)
        sample_indices = np.random.choice(cluster_indices, size=sample_size, replace=True)
        # print(sample_indices)
        # print(X[sample_indices])
        sampled_x.append(X[sample_indices])

        sampled_y.append(df(sample_indices, G, node_order))
        sampled_nodes.append([node_order[idx] for idx in sample_indices])
        # print(i)
        # print(sampled_x)
        # print(sampled_y)
        # print(sampled_nodes)

    sampled_x = np.vstack(sampled_x)
    sampled_y = np.vstack(sampled_y)

    # 示例使用
    original_array = np.array(sampled_y)  # 原始2×3二维数组
    sampled_y = convert_to_N_by_1(original_array)

    # print("经过vstack之后")
    # print(sampled_x.shape)
    # print(sampled_y.shape)
    # print(sampled_x)
    # print(sampled_y)
    # for i, nodes in enumerate(sampled_nodes):
    #     print(f"Sampled nodes from cluster {i + 1}: {nodes}")
    # Step 4: Train the SVR model
    svr_model = SVR(kernel='rbf', C=100, gamma='scale')
    svr_model.fit(sampled_x, sampled_y.ravel())

    # # 遍历X的每一行并进行预测
    # predict_vitality = []
    # for i in range(X.shape[0]):
    #     node_embedding = X[i].reshape(1, -1)  # 取出每一行作为单独的输入，reshape为2D数组
    #     # print(X[i])
    #     predicted_value = svr_model.predict(node_embedding)
    #     # print(predicted_value)
    #     predict_vitality.append(predicted_value[0])  # 预测结果是1D数组，取第一个元素
    #
    # print("图中所有节点的预测活力值是：", predict_vitality)

    # Step 5: Predict node vitality using SVR and neighbor influence

    # predict_vitality = predict_vitality(G, svr_model, X, node_order)
    #
    # predict_vitality = svr_model.predict(X)
    # print("图中所有节点预测活力值是：",predict_vitality)
    sorted_nodes, EML_values = score(G, svr_model, X, node_order, alpha=0.25)
    # 输出
    # Output top 10 sorted nodes based on EML
    # print("Top 10 nodes sorted by EML values (from highest to lowest):")
    # for rank, node in enumerate(sorted_nodes[:k]):
    # for rank, node in enumerate(sorted_nodes):
    #     print(f"Rank {rank + 1}: Node {node}, EML = {EML_values[node]:.4f}")

    dc = nx.degree_centrality(G)
    bc = nx.betweenness_centrality(G)
    cc = nx.closeness_centrality(G)
    ec = nx.eigenvector_centrality(G, max_iter=1000)
    pr = nx.pagerank(G)
    kc = nx.katz_centrality(G, alpha=0.01, max_iter=10000)
    ks = nx.core_number(G)


    sir_rank = [v for n, v in sorted(SIR_model_spreadability.items(), key=lambda x: x[0])]
    # print(sir_rank)
    # print(sorted(G.nodes))
    eml_rank = [-1 * sorted_nodes.index(n) for n in sorted(G.nodes)]
    print(eml_rank)
    print(eml_rank)
    degree_rank = [v for n, v in sorted(dc.items(), key=lambda x: x[0])]
    print([n for n, v in sorted(dc.items(), key=lambda x: x[0])])
    bc_rank = [v for n, v in sorted(bc.items(), key=lambda x: x[0])]
    cc_rank = [v for n, v in sorted(cc.items(), key=lambda x: x[0])]
    ec_rank = [v for n, v in sorted(ec.items(), key=lambda x: x[0])]
    kc_rank = [v for n, v in sorted(kc.items(), key=lambda x: x[0])]
    ks_rank = [v for n, v in sorted(ks.items(), key=lambda x: x[0])]
    pr_rank = [v for n, v in sorted(pr.items(), key=lambda x: x[0])]

    print(f'KNI肯德尔系数: {kendalltau(sir_rank, eml_rank)[0]}')
    print(f'dc肯德尔系数: {kendalltau(sir_rank, degree_rank)[0]}')
    print(f'bc肯德尔系数: {kendalltau(sir_rank, bc_rank)[0]}')
    print(f'cc肯德尔系数: {kendalltau(sir_rank, cc_rank)[0]}')
    print(f'ec肯德尔系数: {kendalltau(sir_rank, ec_rank)[0]}')
    print(f'pr肯德尔系数: {kendalltau(sir_rank, pr_rank)[0]}')
    print(f'kc肯德尔系数: {kendalltau(sir_rank, kc_rank)[0]}')
    print(f'ks肯德尔系数: {kendalltau(sir_rank, ks_rank)[0]}')



    for k in range(10, 151,10):
        top_k_spreadability_nodes = sorted(SIR_model_spreadability, key=SIR_model_spreadability.get, reverse=True)[:k]
        origin_degree_centrality_knodes = sorted(dc, key=dc.get, reverse=True)[:k]
        # origin_degree_centrality = sorted(dc.values(), reverse=True)[:k]
        # print(f'度中心性最高的前{k}个节点分别是{origin_degree_centrality_knodes}')
        origin_betweenness_centrality_knodes = sorted(bc, key=bc.get, reverse=True)[:k]
        # print(f'介数中心性最高的前{k}个节点分别是{origin_betweenness_centrality_knodes}')
        origin_closeness_centrality_knodes = sorted(cc, key=cc.get, reverse=True)[:k]
        # print(f'接近中心性最高的前{k}个节点分别是{origin_closeness_centrality_knodes}')
        origin_eigenvector_centrality_knodes = sorted(ec, key=ec.get, reverse=True)[:k]
        # print(f'特征向量中心性最高的前{k}个节点分别是{origin_eigenvector_centrality_knodes}')
        origin_pagerank_knodes = sorted(pr, key=pr.get, reverse=True)[:k]
        # print(f'pagerank值最高的前{k}个节点分别是{origin_pagerank_knodes}')
        origin_katz_centrality_knodes = sorted(kc, key=kc.get, reverse=True)[:k]
        # print(f'katz中心性最高的前{k}个节点分别是{origin_katz_centrality_knodes}')
        origin_k_shell_knodes = sorted(ks, key=ks.get, reverse=True)[:k]
        # print(f'k_shell值最高的前{k}个节点分别是{origin_k_shell_knodes}')
        origin_sorted_nodes = sorted_nodes[:k]

        #SIR_influence spread
        # print(f'当前轮次的k取值为：{k}')
        # degree_centrality_knodes_spread = SIR_model(G, origin_degree_centrality_knodes,beta=beta, gamma=0.01, num_simulations=1)
        # print(f"度中心性最高的前{k}个节点的最终传播规模是: {degree_centrality_knodes_spread}")
        # betweenness_centrality_knodes_spread = SIR_model(G, origin_betweenness_centrality_knodes,beta=beta, gamma=0.01, num_simulations=1)
        # print(f"介数中心性最高的前{k}个节点的最终传播规模是: {betweenness_centrality_knodes_spread}")
        # closeness_centrality_knodes_spread =SIR_model(G, origin_closeness_centrality_knodes,beta=beta, gamma=0.01, num_simulations=1)
        # print(f"接近中心性最高的前{k}个节点的最终传播规模是: {closeness_centrality_knodes_spread}")
        # eigenvector_centrality_knodes_spread =SIR_model(G, origin_eigenvector_centrality_knodes,beta=beta, gamma=0.01, num_simulations=1)
        # print(f"特征向量中心性最高的前{k}个节点的最终传播规模是: {eigenvector_centrality_knodes_spread}")
        # pagerank_knodes_spread =SIR_model(G, origin_pagerank_knodes,beta=beta, gamma=0.01, num_simulations=1)
        # print(f"PageRank值最高的前{k}个节点的最终传播规模是: {pagerank_knodes_spread}")
        # katz_centrality_knodes_spread =SIR_model(G,origin_katz_centrality_knodes ,beta=beta, gamma=0.01, num_simulations=1)
        # print(f"Katz中心性最高的前{k}个节点的最终传播规模是: {katz_centrality_knodes_spread}")
        # k_shell_knodes_spread =SIR_model(G, origin_k_shell_knodes,beta=beta, gamma=0.01, num_simulations=1)
        # print(f"kshell值最高的前{k}个节点的最终传播规模是: {k_shell_knodes_spread}")
        # sorted_nodes_spread =SIR_model(G, origin_sorted_nodes,beta=beta, gamma=0.01, num_simulations=1)
        # print(f"SDNE_KNI值最高的前{k}个节点的最终传播规模是: {sorted_nodes_spread}")


        #IC_influence spread
        print(f'当前轮次的k取值为：{k}')
        DC2 = IC2(G, origin_degree_centrality_knodes, mc=1000)
        print(f'原始度中心性最高的前{k}个节点作为种子集传播的最终扩展度{DC2}')
        BC2 = IC2(G, origin_betweenness_centrality_knodes, mc=1000)
        print(f'原始介数中心性最高的前{k}个节点作为种子集传播的最终扩展度{BC2}')
        CC2 = IC2(G, origin_closeness_centrality_knodes, mc=1000)
        print(f'原始接近中心性最高的前{k}个节点作为种子集传播的最终扩展度{CC2}')
        EC2 = IC2(G, origin_eigenvector_centrality_knodes, mc=1000)
        print(f'原始特征向量中心性最高的前{k}个节点作为种子集传播的最终扩展度{EC2}')
        PR2 = IC2(G, origin_pagerank_knodes, mc=1000)
        print(f'原始pagerank值最高的前{k}个节点作为种子集传播的最终扩展度{PR2}')
        KT = IC2(G,origin_katz_centrality_knodes,mc=1000)
        print(f"Katz中心性最高的前{k}个节点作为种子集传播的最终扩展度: {KT}")
        KSHELL = IC2(G, origin_k_shell_knodes,mc=1000)
        print(f"kshell值最高的前{k}个节点作为种子集传播的最终扩展度: {KSHELL}")
        KNI = IC2(G, origin_sorted_nodes,mc=1000)
        print(f"SDNE_KNI值最高的前{k}个节点的最终传播规模是: {KNI}")

    for k in range(10, 300, 20):
        top_k_spreadability_nodes = sorted(SIR_model_spreadability, key=SIR_model_spreadability.get, reverse=True)[:k]
        origin_degree_centrality_knodes = sorted(dc, key=dc.get, reverse=True)[:k]
        # origin_degree_centrality = sorted(dc.values(), reverse=True)[:k]
        # print(f'度中心性最高的前{k}个节点分别是{origin_degree_centrality_knodes}')
        origin_betweenness_centrality_knodes = sorted(bc, key=bc.get, reverse=True)[:k]
        # print(f'介数中心性最高的前{k}个节点分别是{origin_betweenness_centrality_knodes}')
        origin_closeness_centrality_knodes = sorted(cc, key=cc.get, reverse=True)[:k]
        # print(f'接近中心性最高的前{k}个节点分别是{origin_closeness_centrality_knodes}')
        origin_eigenvector_centrality_knodes = sorted(ec, key=ec.get, reverse=True)[:k]
        # print(f'特征向量中心性最高的前{k}个节点分别是{origin_eigenvector_centrality_knodes}')
        origin_pagerank_knodes = sorted(pr, key=pr.get, reverse=True)[:k]
        # print(f'pagerank值最高的前{k}个节点分别是{origin_pagerank_knodes}')
        origin_katz_centrality_knodes = sorted(kc, key=kc.get, reverse=True)[:k]
        # print(f'katz中心性最高的前{k}个节点分别是{origin_katz_centrality_knodes}')
        origin_k_shell_knodes = sorted(ks, key=ks.get, reverse=True)[:k]
        # print(f'k_shell值最高的前{k}个节点分别是{origin_k_shell_knodes}')
        origin_sorted_nodes = sorted_nodes[:k]

        # 计算 Jaccard 相似系数
        print(f'当前轮次的k取值为：{k}')
        jaccard_index = jaccard_coefficient(top_k_spreadability_nodes, origin_degree_centrality_knodes)
        print(f"度中心性最高的前{k}个节点的Jaccard相似系数是: {jaccard_index:.4f}")
        jaccard_index = jaccard_coefficient(top_k_spreadability_nodes, origin_betweenness_centrality_knodes)
        print(f"介数中心性最高的前{k}个节点的Jaccard相似系数是: {jaccard_index:.4f}")
        jaccard_index = jaccard_coefficient(top_k_spreadability_nodes, origin_closeness_centrality_knodes)
        print(f"接近中心性最高的前{k}个节点的Jaccard相似系数是: {jaccard_index:.4f}")
        jaccard_index = jaccard_coefficient(top_k_spreadability_nodes, origin_eigenvector_centrality_knodes)
        print(f"特征向量中心性最高的前{k}个节点的Jaccard相似系数是: {jaccard_index:.4f}")
        jaccard_index = jaccard_coefficient(top_k_spreadability_nodes, origin_pagerank_knodes)
        print(f"PageRank值最高的前{k}个节点的Jaccard相似系数是: {jaccard_index:.4f}")
        jaccard_index = jaccard_coefficient(top_k_spreadability_nodes, origin_katz_centrality_knodes)
        print(f"Katz中心性最高的前{k}个节点的Jaccard相似系数是: {jaccard_index:.4f}")
        jaccard_index = jaccard_coefficient(top_k_spreadability_nodes, origin_k_shell_knodes)
        print(f"kshell值最高的前{k}个节点的Jaccard相似系数是: {jaccard_index:.4f}")
        jaccard_index = jaccard_coefficient(top_k_spreadability_nodes, origin_sorted_nodes)
        print(f"SDNE_KNI选出的最高的前{k}个节点的Jaccard相似系数是: {jaccard_index:.4f}")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")






