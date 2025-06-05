import sys
import networkx as nx
import random
import os
import logging
import numpy as np
import gensim
from torch_geometric.datasets import KarateClub
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep

def random_walk(G, v, l):
    """
    游走采样：具体的游走规则是：每次从当前节点的邻居节点中选择一个节点作为下一个节点，选择概率与邻居节点的度成反比；如果有多个候选节点，则选择度最大的那个作为下一个节点。如果当前节点没有邻居节点或没有候选节点，则结束游走。
    G: 图    v: 游走的根节点    l: 游走的序列的长度
    """
    walk_seq = [v]
    while len(walk_seq) < l:
        node = walk_seq[-1]
        neighbors = list(G.neighbors(node))
        if len(neighbors) > 0:
            candidates = []
            for s in neighbors:
                if s not in walk_seq and G.degree(s) > 0:  # 检查度是否大于0
                    m = 1 / G.degree(s)
                    if np.random.uniform(0, 1) < m:
                        candidates.append(s)
            if not candidates:
                break
            # 从candidates中选择一个度最大的节点
            max_out_degree = -1
            max_out_degree_node = None
            for candidate in candidates:
                degree = G.degree(candidate)
                if degree > max_out_degree:
                    max_out_degree = degree
                    max_out_degree_node = candidate

            walk_seq.append(max_out_degree_node)
        else:
            break
    return walk_seq

# def random_walk(G, v, l):
#     """
#     游走采样
#     G: 图    v: 游走的根节点    l: 游走的序列的长度
#     """
#     walk_seq = [v]
#     while len(walk_seq) < l:
#         node = walk_seq[-1]
#         neighbors = list(G.neighbors(node))
#         if len(neighbors) > 0:
#             candidates = []
#             for s in neighbors:
#                 if s not in walk_seq and G.degree(s) > 0:
#                     m = 1 / G.degree(s)
#                     if np.random.uniform(0, 1) < m:
#                         candidates.append(s)
#
#             if candidates:  # 如果存在候选节点
#                 # 随机选择一个候选节点
#                 next_node = random.choice(candidates)
#                 walk_seq.append(next_node)
#             else:
#                 break  # 如果没有候选节点，则跳出循环
#         else:
#             break  # 如果当前节点没有邻居节点，则跳出循环
#
#     return walk_seq


def deep_walk(G, t, d, w, epochs=5):
    """
    深度随机游走。
    G: 图
    t: 每个节点随机游走的次数
    l: 随机游走最大的序列长度
    d: 节点嵌入向量的维度
    w: skipgram的窗口大小
    epochs: skipgram: 训练轮数
    """
    walk_seq_list = []
    nodes = list(G.nodes)
    for walk_length in walk_lengths:
        logger.info(f'当前最大游走序列长度为{walk_length}。')
        logger.info(f'进行采样游走序列。')
        for i in range(t):
            random.shuffle(nodes)
            for node in nodes:
                walk_seq = random_walk(G, node, walk_length)
                walk_seq_list.append(walk_seq)

    logger.info('游走序列采样完成。开始训练skipgram模型...')
    model = gensim.models.Word2Vec(sentences=walk_seq_list, vector_size=d, window=w, max_vocab_size=len(G), min_count=0,
                                   epochs=epochs, sg=1, hs=0)
    return model


# 使用余弦相似度
# def get_k_max_influence_nodes(G, k, model):
#     """
#     获取影响力最大的k个节点。
#     G: 图
#     k: 获取的节点数
#     model: 训练的skipgram模型
#     """
#     node_similar_k_nodes_dict = {}
#     with tqdm(total=k, desc=f'计算每个节点最相似的{k}个节点，并统计每个节点出现次数') as tbar:
#         for i, node in enumerate(G.nodes, 1):
#             for sim_node, _ in model.wv.similar_by_key(node, topn=k):
#                 if sim_node not in node_similar_k_nodes_dict:
#                     node_similar_k_nodes_dict[sim_node] = 1
#                 else:
#                     node_similar_k_nodes_dict[sim_node] += 1
#             tbar.set_postfix_str(f'剩余{G.number_of_nodes() - i - 1}个节点')
#             tbar.update(1)
#     sorted_nodes_by_freq = sorted(node_similar_k_nodes_dict.items(), key=lambda x: x[1], reverse=True)[:k]
#     return np.array(sorted_nodes_by_freq)[:, 0], sorted_nodes_by_freq


def get_k_max_influence_nodes(G, k, model):

    node_influence_dict = {}  # 存储每个节点的综合影响

    with tqdm(total=G.number_of_nodes(), desc=f'计算每个节点的综合影响') as tbar:
        for i, node in enumerate(G.nodes, 1):
            node_influence = 0  # 初始化当前节点的综合影响为0
            for sim_node in G.nodes:
                    # 计算两个节点的皮尔逊相关系数
                pearson_corr, _ = pearsonr(model.wv[node], model.wv[sim_node])
                node_influence += pearson_corr  # 累加皮尔逊相关系数
            node_influence /= (G.number_of_nodes())  # 计算平均值
            node_influence_dict[node] = node_influence# 存储当前节点的综合影响
            tbar.set_postfix_str(f'剩余{G.number_of_nodes() - i - 1}个节点')
            tbar.update(1)

            # 按照综合影响进行排序
    sorted_nodes_by_influence = sorted(node_influence_dict.items(), key=lambda x: x[1], reverse=True)[:k]

    return np.array(sorted_nodes_by_influence)[:, 0], sorted_nodes_by_influence

def IC(g, S, mc):
    """
        传播
        传参：图，种子集合，传播次数
        """
    spread = {}
    for j in range(mc):
        new_active, A = S[:], S[:]
        while new_active:
            new_ones = []
            for node in new_active:
                for s in g.neighbors(node):
                    if s not in S:

                        m = 1 / g.degree(s)

                        if np.random.uniform(0, 1) < m:
                            new_ones.append(s)

            new_active = list(set(new_ones) - set(A))
            A += new_active
        spread = set(A)
    return spread


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


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    # 创建一个handler
    handler = logging.StreamHandler()
    logger.setLevel(logging.DEBUG)  # 设置日志级别为DEBUG，这样所有级别的日志都会被处理
    # 创建一个格式器，并添加到handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # 给logger添加handler
    logger.addHandler(handler)

    network_name_dict = { '0':'dolphins','1': 'karateclub', '2': 'jazz', '3': 'facebook', '4': 'lastfm_asia_edges','5':'Europe','6':'europe'}
    network_name_key = input(
        '请输入选择使用的网络编号: 0:dolphins,1:katareclub,2:jazz,3:facebook,4:lastmf_asia ,5:Europe,6:europe')
    if network_name_key not in network_name_dict:
        logger.error('请输入0或1或2或3或4或5或6')
        sys.exit(0)
    network_name = network_name_dict[network_name_key]

    if network_name == 'dolphins':
        data = pd.read_csv('data/dolphins.csv', sep=',', header=None)
    elif network_name == 'karateclub':
        data = pd.read_csv('data/karate.txt', sep='\t', header=None)
    elif network_name == 'jazz':
        data = pd.read_csv('data/jazz.txt', sep=' ', header=None)
    elif network_name == 'facebook':
        data = pd.read_csv('data/facebook_combined.txt', sep=' ', header=None)
    elif network_name == 'lastfm_asia_edges':
        data = pd.read_csv('data/lastfm_asia_edges.csv', sep=',', header=None)
    elif network_name == 'Europe':
        data = pd.read_csv('data/europe-airports.edgelist', sep=' ', header=None)
    elif network_name == 'europe':
        data = pd.read_csv('data/europe-airports.edgelist', sep=' ', header=None)

    data.columns = ['source', 'target']
    data['source'] = data['source'].astype(pd.StringDtype())
    data['target'] = data['target'].astype(pd.StringDtype())
    G = nx.from_pandas_edgelist(data)

    logger.info(f'使用{network_name}数据集，图有{G.number_of_nodes()}个节点，{G.number_of_edges()}条边。')
    # use_model = True
    use_model = False
    w, d, t,  = 4, 64, 10
    k = 35
    Z = []
    for i in range(10):
        logger.info(f'当前是第{i}轮。')
        if use_model and os.path.exists(f'{network_name}.model'):
            logger.info(f'使用已训练完成模型{network_name}.model。')
            model = gensim.models.Word2Vec.load(f'{network_name}.model')
        else:
            logger.info('进行游走采样...')

            # DOLPHINS:
            # walk_lengths = [3,6,10,14,18,22,26,30,34]
            walk_lengths = [35]
            # karateclub:
            # walk_lengths = [3,6,10,13,17,20,24]
            #jazz
            # walk_lengths = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52]
            # walk_lengths = [4,8,12,16,20,24,28,32,36,40,44,48,52,54,60]
            # walk_lengths = [4, 8, 12, 16, 20, 24, 28, 32, 36,40,44,48,52,56,60]
            # walk_lengths = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50,55,60]
            #Europe
            # walk_lengths = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40,44,48]
            # walk_lengths = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50,55,60,70,80]
            # walk_lengths = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

            model = deep_walk(G, t, d, w)
            model.save(f'{network_name}.model')
            logger.info('训练完成，并保'
                        '存模型。')


        topk_nodes, sorted_nodes_by_freq = get_k_max_influence_nodes(G, k, model)
        # logger.info(f'前{k}个节点出现的次数{sorted_nodes_by_freq}。')
        S2 = np.array(sorted_nodes_by_freq)[:, 0]
        S2_list = S2.tolist()
        logger.info(f'前{k}个节点分别是{S2_list}。')

        mc = 1000
        Zi = IC2(G, S2_list, mc)
        Z.append(Zi)
        logger.info(f'种子集的最终传播范围是{Zi}。')
    logger.info(f'种子集传播的扩展度分别是{Z}。')

    #
    # dc = nx.degree_centrality(G)
    # # 原始节点的度中心性最大前k个
    # origin_degree_centrality = sorted(dc.values(), reverse=True)[:k]
    # origin_degree_centrality_knodes = sorted(dc, key=dc.get, reverse=True)[:k]
    # logger.info(f'原始度中心性最高的前{k}个节点分别是{origin_degree_centrality_knodes}')
    # DC2 = IC2(G, origin_degree_centrality_knodes, mc)
    # logger.info(f'原始度中心性最高的前{k}个节点作为种子集传播的最终扩展度{DC2}。')
    #
    #
    #  # 所选择的前k个种子节点的接近中心性
    # cc = nx.closeness_centrality(G)
    # origin_closeness_centrality = sorted(cc.values(), reverse=True)[:k]
    # origin_closeness_centrality_knodes = sorted(cc, key=cc.get, reverse=True)[:k]
    # logger.info(f'原始接近中心性最高的前{k}个节点分别是{origin_closeness_centrality_knodes}')
    # CC2 = IC2(G, origin_closeness_centrality_knodes, mc)
    # logger.info(f'原始接近中心性最高的前{k}个节点作为种子集传播的最终扩展度{CC2}。')
    #
    # # 所选择的前k个种子节点的特征向量中心性
    # ec = nx.eigenvector_centrality(G,max_iter=1000)
    # origin_eigenvector_centrality = sorted(ec.values(), reverse=True)[:k]
    # origin_eigenvector_centrality_knodes = sorted(ec, key=ec.get, reverse=True)[:k]
    # logger.info(f'原始特征向量中心性最高的前{k}个节点分别是{origin_eigenvector_centrality_knodes}')
    # EC2 = IC2(G, origin_eigenvector_centrality_knodes, mc)
    # logger.info(f'原始特征向量中心性最高的前{k}个节点作为种子集传播的最终扩展度{EC2}。')
    #
    #
    #
    # # 所选择的前k个种子节点的PageRank值
    # pr = nx.pagerank(G)
    # origin_pagerank = sorted(pr.values(), reverse=True)[:k]
    # origin_pagerank_knodes = sorted(pr, key=pr.get, reverse=True)[:k]
    # logger.info(f'原始pagerank值最高的前{k}个节点分别是{origin_pagerank_knodes}')
    # PR2 = IC2(G, origin_pagerank_knodes, mc)
    # logger.info(f'原始pagerank值最高的前{k}个节点作为种子集传播的最终扩展度{PR2}。')
    #

