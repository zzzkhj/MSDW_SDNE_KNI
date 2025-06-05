# import random
# import networkx as nx
# import time
#
# def SIR_model_spreadability(G, beta=0.03, gamma=0.01, iterations=1000):
#     # 存储每个节点的传播能力
#     spreadability = {}
#
#     # 遍历图中每个节点作为初始感染节点
#     for initial_infected in G.nodes():
#         S = set(G.nodes())  # Susceptible
#         I = {initial_infected}  # Infected
#         R = set()  # Recovered
#         S.remove(initial_infected)
#         total_infected = set(I)  # 用于记录所有被感染的节点
#
#         for _ in range(iterations):
#             new_infected = set()
#             new_recovered = set()
#
#             # 遍历当前感染节点
#             for node in I:
#                 # 感染邻居
#                 neighbors = G.neighbors(node)
#                 for neighbor in neighbors:
#                     if neighbor in S and random.random() < beta:
#                         new_infected.add(neighbor)
#
#                 # 节点可能康复
#                 if random.random() < gamma:
#                     new_recovered.add(node)
#
#             # 更新状态
#             S -= new_infected
#             I = (I - new_recovered) | new_infected
#             R |= new_recovered
#
#             # 记录总感染节点
#             total_infected |= new_infected
#
#             # 如果没有新的感染者，结束传播
#             if len(I) == 0:
#                 break
#
#         # 计算该节点的传播能力
#         spreadability[initial_infected] = len(total_infected) / G.number_of_nodes()
#
#     return spreadability
#
# if __name__ == "__main__":
#     start_time = time.time()
#        # Load graph
#     G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',
#                          create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
#
#     result = SIR_model_spreadability(G)
#     print(result)
#     end_time = time.time()
#     print(f"Total execution time: {end_time - start_time:.2f} seconds")

import random
import networkx as nx
import time
import numpy as np
# from sklearn.metrics import jaccard_score
from scipy.stats import kendalltau


def SIR_model_spreadability(G, beta=0.042, gamma=0.01, iterations=50, num_simulations=1):
    # 存储每个节点的传播能力的平均值
    spreadability = {node: 0 for node in G.nodes()}

    # 遍历图中每个节点作为初始感染节点
    for initial_infected in G.nodes():
        total_spread = 0  # 累积传播能力

        # 对每个节点进行1000次（num_simulations次）SIR模型模拟
        for _ in range(num_simulations):
            S = set(G.nodes())  # Susceptible
            I = {initial_infected}  # Infected
            R = set()  # Recovered
            S.remove(initial_infected)
            total_infected = set(I)  # 用于记录所有被感染的节点

            for _ in range(iterations):
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

                # 如果没有新的感染者，结束传播
                if len(I) == 0:
                    break

            # 记录当前模拟中的传播能力（感染到的节点数）
            total_spread += len(total_infected) / G.number_of_nodes()

        # 计算该节点的平均传播能力
        spreadability[initial_infected] = total_spread / num_simulations

    return spreadability
def calculate_jaccard_and_kendall(SIR_sorted_nodes, algo_nodes, k_range):
    jaccard_scores = []
    kendall_tau_scores = []

    # 打印 SIR_sorted_nodes 的内容和结构
    print("SIR_sorted_nodes:", SIR_sorted_nodes)
    print("Sample of SIR_sorted_nodes:", SIR_sorted_nodes[:5])  # 打印前5个元素

    for k in k_range:
        # 确保不会超出 SIR_sorted_nodes 的长度
        k = min(k, len(SIR_sorted_nodes), len(algo_nodes))

        # 确保解包正确
        try:
            SIR_top_k_list = [node for node, _ in SIR_sorted_nodes[:k]]
        except ValueError as e:
            print(f"Error while unpacking SIR_sorted_nodes at k={k}: {e}")
            continue  # 跳过当前 k 值的处理

        algo_top_k_list = algo_nodes[:k]  # 假设 algo_nodes 是一个列表

        # 检查两个列表的长度是否一致
        if len(SIR_top_k_list) == len(algo_top_k_list):
            # 计算 Jaccard 系数
            intersection = len(set(SIR_top_k_list) & set(algo_top_k_list))
            union = len(set(SIR_top_k_list) | set(algo_top_k_list))
            jaccard_index = intersection / union if union != 0 else 0
            jaccard_scores.append(jaccard_index)

            # 计算 Kendall's Tau
            tau, _ = kendalltau(SIR_top_k_list, algo_top_k_list)
            kendall_tau_scores.append(tau)
        else:
            # 输出警告信息
            print(f"Warning: Different lengths for k={k}: SIR({len(SIR_top_k_list)}), Algo({len(algo_top_k_list)})")

    return jaccard_scores, kendall_tau_scores


if __name__ == "__main__":
    start_time = time.time()

    # Load graph
    G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',
                         create_using=nx.Graph(), nodetype=None, data=[('weight', int)])

    # 运行 SIR 模型，并获得传播能力
    result = SIR_model_spreadability(G)

    # 对节点按传播能力排序
    sorted_spreadability = sorted(result.items(), key=lambda x: x[1], reverse=True)

    # 输出排名结果
    for rank, (node, spread) in enumerate(sorted_spreadability, start=1):
        print(f"Rank {rank}: Node {node} with spreadability {spread:.4f}")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    dc = nx.degree_centrality(G)
    # 原始节点的度中心性最大前k个
    origin_degree_centrality = sorted(dc.values(), reverse=True)[:20]
    origin_degree_centrality_knodes = sorted(dc, key=dc.get, reverse=True)[:20]
    print(f'原始度中心性最高的前{20}个节点分别是{origin_degree_centrality_knodes}')

     # 所选择的前k个种子节点的接近中心性
    cc = nx.closeness_centrality(G)
    origin_closeness_centrality = sorted(cc.values(), reverse=True)[:20]
    origin_closeness_centrality_knodes = sorted(cc, key=cc.get, reverse=True)[:20]
    print(f'原始接近中心性最高的前{20}个节点分别是{origin_closeness_centrality_knodes}')

    # 所选择的前k个种子节点的特征向量中心性
    ec = nx.eigenvector_centrality(G,max_iter=1000)
    origin_eigenvector_centrality = sorted(ec.values(), reverse=True)[:20]
    origin_eigenvector_centrality_knodes = sorted(ec, key=ec.get, reverse=True)[:20]
    print(f'原始特征向量中心性最高的前{20}个节点分别是{origin_eigenvector_centrality_knodes}')

    # 所选择的前k个种子节点的PageRank值
    pr = nx.pagerank(G)
    origin_pagerank = sorted(pr.values(), reverse=True)[:20]
    origin_pagerank_knodes = sorted(pr, key=pr.get, reverse=True)[:20]
    print(f'原始pagerank值最高的前{20}个节点分别是{origin_pagerank_knodes}')
# SIR模型的节点传播性排序
    SIR_sorted_nodes = [node for node, _ in sorted_spreadability]

    # 各算法生成的前20个节点排序
    algo_degree_nodes = origin_degree_centrality_knodes
    algo_closeness_nodes = origin_closeness_centrality_knodes
    algo_eigenvector_nodes = origin_eigenvector_centrality_knodes
    algo_pagerank_nodes = origin_pagerank_knodes

    # 设置 k 的范围， [10-290] 之间步进为10
    k_range = range(10, 300, 10)

    # 计算每种算法的Jaccard和Kendall's Tau
    for algo_name, algo_nodes in [
        ('Degree Centrality', algo_degree_nodes),
        ('Closeness Centrality', algo_closeness_nodes),
        ('Eigenvector Centrality', algo_eigenvector_nodes),
        ('PageRank', algo_pagerank_nodes),
    ]:
        # jaccard_scores, kendall_tau_scores = calculate_jaccard_and_kendall(SIR_sorted_nodes, algo_nodes, k_range)
        # 对节点按传播能力排序
        sorted_spreadability = sorted(result.items(), key=lambda x: x[1], reverse=True)

        # 调用计算 Jaccard 和 Kendall 的函数
        jaccard_scores, kendall_tau_scores = calculate_jaccard_and_kendall(sorted_spreadability, algo_nodes, k_range)
        # 输出结果
        print(f"--- {algo_name} ---")
        for k, jaccard, tau in zip(k_range, jaccard_scores, kendall_tau_scores):
            print(f"Top-{k} Jaccard: {jaccard:.4f}, Kendall's Tau: {tau:.4f}")
