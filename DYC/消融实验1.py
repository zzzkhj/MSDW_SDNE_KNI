import sys
import networkx as nx
import random
import os
import logging
import numpy as np
import gensim

import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr

#针对两个小数据集 去掉模拟传播即随便选节点随机游走 即deepwalk方法 后面还是用皮尔逊相似系数


def random_walk(G, v, l):
    """
    随机游走。
    G: 图
    v: 游走的根节点
    l: 游走的序列的长度
    """
    walk_seq = [v]
    while len(walk_seq) < l:
        node = walk_seq[-1]
        neighbors = list(G.neighbors(node))
        if len(neighbors) > 0:
            walk_seq.append(random.choice(neighbors))
        else:
            break


    return walk_seq

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
    w, d, t,  = 4, 64, 6
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
            walk_lengths = [3,6,10,14,18,22,26,30,34]
            # walk_lengths = [35]
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
            logger.info('训练完成，并保存模型。')


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


