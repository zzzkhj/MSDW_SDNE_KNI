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

def random_walk(G, v, l):
    """
    游走采样：具体的游走规则是：每次从当前节点的邻居节点中选择一个节点作为下一个节点，选择概率与邻居节点的入度成反比；如果有多个候选节点，则选择出度最大的那个作为下一个节点。如果当前节点没有邻居节点或没有候选节点，则结束游走。
    G: 图    v: 游走的根节点    l: 游走的序列的长度
    """
    walk_seq = [v]
    while len(walk_seq) < l:
        node = walk_seq[-1]
        neighbors = list(G.neighbors(node))
        if len(neighbors) > 0:
            candidates = []
            for s in neighbors:
                if s not in walk_seq and G.in_degree(s) > 0:  # 检查入度是否大于0
                    m = 1 / G.in_degree(s)
                    if np.random.uniform(0, 1) < m:
                        candidates.append(s)

                        # 如果candidates为空，则终止游走或采取其他策略
            if not candidates:
                break  # 或者可以选择一个已经访问过的节点，或者重新开始游走等。

            # 从candidates中选择一个出度最大的节点
            max_out_degree = -1
            max_out_degree_node = None
            for candidate in candidates:
                out_degree = G.out_degree(candidate)
                if out_degree > max_out_degree:
                    max_out_degree = out_degree
                    max_out_degree_node = candidate

            walk_seq.append(max_out_degree_node)
        else:
            # 没有邻居节点时，可以选择终止游走或采取其他策略
            break  # 或者可以选择一个已经访问过的节点等。

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
#                 if s not in walk_seq and G.in_degree(s) > 0:
#                     m = 1 / G.in_degree(s)
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


# dyc change
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
            node_influence_dict[node] = node_influence
            tbar.set_postfix_str(f'剩余{G.number_of_nodes() - i - 1}个节点')# 存储当前节点的综合影响
            tbar.update(1)

            # 按照综合影响进行排序
    sorted_nodes_by_influence = sorted(node_influence_dict.items(), key=lambda x: x[1], reverse=True)[:k]

    return np.array(sorted_nodes_by_influence)[:, 0], sorted_nodes_by_influence

def plot_side_by_side_bar_charts(datasets, categories, title, bar_width=0.4, colors=None, labels=None):
    """
    绘制并列的柱状图。
    参数:
    datasets (list of lists): 包含多个数据集的列表，每个数据集是一个数值列表。
    categories (list): 类别标签列表，用于x轴。
    title（str）：图的标题
    bar_width (float): 每个柱子的宽度。
    colors (list of str): 可选参数，用于指定每个数据集的颜色。
    labels (list of str): 可选参数，用于指定每个数据集的标签。
    """
    # 确保数据集、类别、颜色和标签的数量匹配
    num_datasets = len(datasets)
    if colors is not None and len(colors) != num_datasets:
        raise ValueError("Number of colors must match the number of datasets.")
    if labels is not None and len(labels) != num_datasets:
        raise ValueError("Number of labels must match the number of datasets.")

        # 计算x坐标位置
    x_positions = np.arange(len(categories))
    offset = bar_width
    x_positions_all = [x_positions + i * offset for i in range(num_datasets)]

    plt.title(title)
    # 绘制每个数据集的柱状图
    for i in range(num_datasets):
        color = colors[i] if colors is not None else 'C{}'.format(i)
        label = labels[i] if labels is not None else 'Dataset {}'.format(i + 1)
        plt.bar(x_positions_all[i], datasets[i], width=bar_width, label=label, color=color)

        # 添加图例
    plt.legend()

    # 设置x轴和y轴标签
    plt.xlabel('Node Id')
    plt.ylabel('Values')

    # 设置x轴刻度位置和标签
    plt.xticks([x + bar_width * (num_datasets - 0.5) / 2 for x in x_positions], categories)


def IC(g, S, mc):
    spread = {}

    for j in range(mc):
        new_active, A = S[:], S[:]
        while new_active:
            new_ones = []
            for node in new_active:

                for s in g.neighbors(node):
                    if s not in S:
                        # 第一种方法：获得邻居结点s的入度分之一
                        m = 1 / g.degree(s)
                        # 第二种：获得邻居结点的联系次数分之一

                        if np.random.uniform(0, 1) < m:
                            new_ones.append(s)
                            # 对结点和邻居结点之间的联系时间进行排序

            new_active = list(set(new_ones) - set(A))
            A += new_active
        spread = set(A)
    return spread


def IC2(g, S, mc):
    """
    传播
    传参：图，种子集合，传播次数
    """
    spread = []

    for j in range(mc):
        new_active, A = S[:], S[:]
        while new_active:
            new_ones = []
            for node in new_active:
                for s in g.neighbors(node):
                    if s not in S:
                        # 第一种方法：获得邻居结点s的入度分之一
                        m = 1 / g.in_degree(s)
                        if np.random.uniform(0, 1) < m:
                            new_ones.append(s)

            new_active = list(set(new_ones) - set(A))
            A += new_active

        spread.append(len(A))
    return np.mean(spread)

# 生成随机有向图
def generate_random_directed_graph(num_nodes, p):
    # 使用类似Erdős-Rényi模型的gnp_random_graph生成随机有向图
    # p是每条边出现的概率
    G_directed = nx.gnp_random_graph(num_nodes, p, directed=True)
    return G_directed
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

    network_name_dict = { '1': 'epinions', '2': 'Wiki-Vote', '3': 'congress',
                         '4': 'p2p-Gnutella08', '5': 'email-Eu-core','6':'ca_grqc','7':'soc-sign-bitcoin-alpha','8':'soc-sign-bitcoin-otc','9':'p2p_gnutella09','10':'p2p_gnutella05','11':'p2p_gnutella06','12':'G_directed'}
    network_name_key = input(
        '请输入选择使用的网络编号: 1：epinions：，2：Wiki-Vote, 3:congress, 4:p2p-Gnutella08,5:email-Eu-core , 6:ca_grqc,7: soc-sign-bitcoin-alpha,8:soc-sign-bitcoin-otc,9:p2p_gnutella09,10:p2p_gnutella05,11:p2p_gnutella06,12:G_directed')
    if network_name_key not in network_name_dict:
        logger.error('请输入1或2或3或4或5或6或7或8或9或10或11')
        sys.exit(0)
    network_name = network_name_dict[network_name_key]

    if network_name == 'epinions':
        data = pd.read_csv('data/soc-Epinions1.txt', sep='\t', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
    elif network_name == 'Wiki-Vote':
        data = pd.read_csv('data/Wiki-Vote.txt', sep='\t', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
    elif network_name == 'congress':
        data = pd.read_csv('data/congress.txt', sep=' ', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
    elif network_name == 'p2p-Gnutella08':
        data = pd.read_csv('data/p2p-Gnutella08.txt', sep='\t', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
    elif network_name == 'email-Eu-core':
        data = pd.read_csv('data/email-Eu-core.txt', sep=' ', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
    elif network_name == 'dolphins':
        data = pd.read_csv('data/dolphins.csv', sep=',', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
    elif network_name == 'college_msg':
        data = pd.read_csv('data/college_msg.txt', sep=' ', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
    elif network_name == 'ca_grqc':
        # data = pd.read_csv('data/soc-Epinions1.txt', sep='\t', header=None)
        data = pd.read_csv('data/CA-GrQc.txt', sep='\t', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
    elif network_name == 'soc-sign-bitcoin-alpha':
        data = pd.read_csv('data/soc-sign-bitcoin-alpha.csv', sep=' ', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
    elif network_name == 'soc-sign-bitcoin-otc':
        data = pd.read_csv('data/soc-sign-bitcoin-otc.csv', sep=' ', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
    elif network_name == 'p2p_gnutella09':
        data = pd.read_csv('data/p2p-Gnutella09.txt', sep='\t', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
    elif network_name == 'p2p_gnutella05':
        data = pd.read_csv('data/p2p-Gnutella05.txt', sep='\t', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
    elif network_name == 'p2p_gnutella06':
        data = pd.read_csv('data/p2p-Gnutella06.txt', sep='\t', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
    elif network_name == 'G_directed':
        G = generate_random_directed_graph(1000, 0.02)


    logger.info(f'使用{network_name}数据集，图有{G.number_of_nodes()}个节点，{G.number_of_edges()}条边。')
    use_model = True
    # use_model = False
    w, d, t,  = 4, 64, 50
    k = 20
    Z = []

    for i in range(3):
        logger.info(f'当前是第{i}轮。')
        if use_model and os.path.exists(f'{network_name}.model'):
            logger.info(f'使用已训练完成模型{network_name}.model。')
            model = gensim.models.Word2Vec.load(f'{network_name}.model')
        else:
            logger.info('进行游走采样...')
            # walk_lengths = [50, 100, 150, 200, 250]
            # walk_lengths = [5,10,15,20,25, 30,35, 40,45,50, 60, 70, 80, 90, 100,110,120,130,140,150]
            #congress
            walk_lengths = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
            #walk_lengths = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150]
            model = deep_walk(G, t, d, w)
            model.save(f'{network_name}.model')
            logger.info('训练完成，并保存模型。')


        topk_nodes, sorted_nodes_by_freq = get_k_max_influence_nodes(G, k, model)
        # logger.info(f'前{k}个节点出现的次数{sorted_nodes_by_freq}。')
        S2 = np.array(sorted_nodes_by_freq)[:, 0]
        S2_list = S2.tolist()
        logger.info(f'前{k}个节点分别是{S2_list}。')
        mc = 1000
        # Z1 = IC(G, S2_list, mc)
        Zi = IC2(G, S2_list, mc)
        Z.append(Zi)
        logger.info(f'种子集的最终传播范围是{Zi}。')
    logger.info(f'种子集传播的扩展度分别是{Z}。')
    # # print(np.array(sorted_nodes_by_freq)[:, 0])

    dc = nx.degree_centrality(G)
    degree_centrality = [dc.get(n) for n in topk_nodes]
    # 原始节点的度中心性最大前k个
    origin_degree_centrality = sorted(dc.values(), reverse=True)[:k]
    origin_degree_centrality_knodes = sorted(dc, key=dc.get, reverse=True)[:k]
    logger.info(f'原始度中心性最高的前{k}个节点分别是{origin_degree_centrality_knodes}')

    # DC1 = IC(G, origin_degree_centrality_knodes, mc)
    DC2 = IC2(G, origin_degree_centrality_knodes, mc)
    # logger.info(f'原始度中心性最高的前{k}个节点作为种子集的最终传播范围是{DC1}。')
    logger.info(f'原始度中心性最高的前{k}个节点作为种子集传播的最终扩展度{DC2}。')

     # 所选择的前k个种子节点的接近中心性
    cc = nx.closeness_centrality(G)
    closeness_centrality = [cc.get(n) for n in topk_nodes]
    origin_closeness_centrality = sorted(cc.values(), reverse=True)[:k]
    origin_closeness_centrality_knodes = sorted(cc, key=cc.get, reverse=True)[:k]
    logger.info(f'原始接近中心性最高的前{k}个节点分别是{origin_closeness_centrality_knodes}')

    # CC1 = IC(G, origin_closeness_centrality_knodes, mc)
    CC2 = IC2(G, origin_closeness_centrality_knodes, mc)
    # logger.info(f'原始接近中心性最高的前{k}个节点作为种子集的最终传播范围是{CC1}。')
    logger.info(f'原始接近中心性最高的前{k}个节点作为种子集传播的最终扩展度{CC2}。')

    # 所选择的前k个种子节点的特征向量中心性
    ec = nx.eigenvector_centrality(G,max_iter=1000)
    eigenvector_centrality = [ec.get(n) for n in topk_nodes]
    origin_eigenvector_centrality = sorted(ec.values(), reverse=True)[:k]
    origin_eigenvector_centrality_knodes = sorted(ec, key=ec.get, reverse=True)[:k]
    logger.info(f'原始特征向量中心性最高的前{k}个节点分别是{origin_eigenvector_centrality_knodes}')

    # EC1 = IC(G, origin_eigenvector_centrality_knodes, mc)
    EC2 = IC2(G, origin_eigenvector_centrality_knodes, mc)
    # logger.info(f'原始特征向量中心性最高的前{k}个节点作为种子集的最终传播范围是{EC1}。')
    logger.info(f'原始特征向量中心性最高的前{k}个节点作为种子集传播的最终扩展度{EC2}。')

    # 所选择的前k个种子节点的PageRank值
    pr = nx.pagerank(G)
    pagerank = [pr.get(n) for n in topk_nodes]
    origin_pagerank = sorted(pr.values(), reverse=True)[:k]
    origin_pagerank_knodes = sorted(pr, key=pr.get, reverse=True)[:k]
    logger.info(f'原始pagerank值最高的前{k}个节点分别是{origin_pagerank_knodes}')

    # PR1 = IC(G, origin_pagerank_knodes, mc)
    PR2 = IC2(G, origin_pagerank_knodes, mc)
    # logger.info(f'原始pagerank值最高的前{k}个节点作为种子集的最终传播范围是{PR1}。')
    logger.info(f'原始pagerank值最高的前{k}个节点作为种子集传播的最终扩展度{PR2}。')

    # logger.info('开始画图...')
    #
    # plt.figure()
    # plot_side_by_side_bar_charts([degree_centrality, closeness_centrality,eigenvector_centrality,pagerank],
    #                              [str(n) for n in topk_nodes], 'centrality', 0.2, ['b', 'g','r','y'], ['DC', 'CC','EC','PR'])
    #
    # plt.figure()
    # plot_side_by_side_bar_charts([origin_degree_centrality, degree_centrality],
    #                              [str(n) for n in range(1, k + 1)], 'degree centrality', 0.2, ['b', 'g'],
    #                              ['Origin Node', 'MI Node'])
    # plt.figure()
    # plot_side_by_side_bar_charts([origin_closeness_centrality, closeness_centrality],
    #                              [str(n) for n in range(1, k + 1)], 'closeness centrality', 0.2, ['b', 'g'],
    #                              ['Origin Node', 'MI Node'])
    # plt.figure()
    # plot_side_by_side_bar_charts([origin_eigenvector_centrality, eigenvector_centrality],
    #                              [str(n) for n in range(1, k + 1)], 'eigenvector_centrality', 0.2, ['b', 'g'],
    #                              ['Origin Node', 'MI Node'])
    # plt.figure()
    # plot_side_by_side_bar_charts([origin_pagerank, pagerank],
    #                              [str(n) for n in range(1, k + 1)], 'pagerank', 0.2, ['b', 'g'],
    #                              ['Origin Node', 'MI Node'])
    # plt.show()
    # logger.info('完成。')

