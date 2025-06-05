import logging
import sys
import numpy as np
import networkx as nx
import pandas as pd
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
                        m = 1 / g.degree(s)

                        if np.random.uniform(0, 1) < m:
                            new_ones.append(s)

            new_active = list(set(new_ones) - set(A))
            A += new_active

        spread.append(len(A))
    return np.mean(spread)


if __name__ == '__main__':
    # walks = simulate_walks(args.num_walks, args.walk_length)
    # print(walks)
    logger = logging.getLogger(__name__)
    # 创建一个handler
    handler = logging.StreamHandler()
    logger.setLevel(logging.DEBUG)  # 设置日志级别为DEBUG，这样所有级别的日志都会被处理
    # 创建一个格式器，并添加到handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(handler)


    k=10
    use_model = False
    # use_model = True
    network_name_dict = {'0': 'KarateClub', '1': 'epinions', '2': 'Wiki-Vote', '3': 'facebook_combined','4': 'congress', '5': 'lastfm_asia_edges', '6': 'jazz','7':'ca_grqc'}
    network_name_key = input('选择使用的网络：0：KarateClub，1：epinions：，2：Wiki-Vote, 3:facebook_combined, 4:congress,5:lastfm_asia_edges, 6:jazz,7:ca_grqc')
    if network_name_key not in network_name_dict:
        logger.error('请输入0或1或2或3或4或5或6')
        sys.exit(0)
    network_name = network_name_dict[network_name_key]

    if network_name == 'KarateClub':
        data = pd.read_csv('data/karate.txt', sep='\t', header=None)
        data.columns = ['source', 'target']

        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())

        # G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
    elif network_name == 'epinions':
        data = pd.read_csv('../MSRW/data/soc-Epinions1.txt', sep='\t', header=None)
        data.columns = ['source', 'target']

        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())

        # G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
    elif network_name == 'Wiki-Vote':

        data = pd.read_csv('data/Wiki-Vote.txt', sep='\t', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(str)
        data['target'] = data['target'].astype(str)

        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())

    elif network_name == 'facebook_combined':

        data = pd.read_csv('data/facebook_combined.txt', sep=' ', header=None)
        data.columns = ['source', 'target']

        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data)

    elif network_name == 'congress':
        # data = pd.read_csv('data/soc-Epinions1.txt', sep='\t', header=None)
        data = pd.read_csv('data/new_data.txt', sep=' ', header=None)
        data.columns = ['source', 'target']

        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())


    elif network_name == 'lastfm_asia_edges':
        # data = pd.read_csv('data/soc-Epinions1.txt', sep='\t', header=None)
        data = pd.read_csv('data/lastfm_asia_edges.csv', sep=',', header=None)
        data.columns = ['source', 'target']

        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data)
    elif network_name == 'jazz':
        # data = pd.read_csv('data/soc-Epinions1.txt', sep='\t', header=None)
        data = pd.read_csv('data/jazz.txt', sep=' ', header=None)
        data.columns = ['source', 'target']

        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data)
    elif network_name == 'ca_grqc':
        # data = pd.read_csv('data/soc-Epinions1.txt', sep='\t', header=None)
        data = pd.read_csv('data/CA-GrQc.txt', sep='\t', header=None)
        data.columns = ['source', 'target']

        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data,create_using=nx.DiGraph())
    logger.info(f'使用{network_name}数据集，图有{G.number_of_nodes()}个节点，{G.number_of_edges()}条边。')



    # print(model.wv.key_to_index)
    # topk_nodes, sorted_nodes_by_freq = get_k_max_influence_nodes(G, k, model)
    # topk_nodes = list(map(int, topk_nodes))
    # logger.info(f'前{k}个节点出现的次数{sorted_nodes_by_freq}。')





    # dc = nx.degree_centrality(G)
    # # degree_centrality = [dc.get(n) for n in topk_nodes]
    # # 原始节点的度中心性最大前k个
    # origin_degree_centrality = sorted(dc.values(), reverse=True)[:k]
    # origin_degree_centrality_knodes = sorted(dc, key=dc.get, reverse=True)[:k]
    # logger.info(f'原始度中心性最高的前{k}个节点分别是{origin_degree_centrality_knodes}')
    #
    # # DC1 = IC(G, origin_degree_centrality_knodes, mc)
    # DC2 = IC2(G, origin_degree_centrality_knodes, mc=1000)
    # # logger.info(f'原始度中心性最高的前{k}个节点作为种子集的最终传播范围是{DC1}。')
    # logger.info(f'原始度中心性最高的前{k}个节点作为种子集传播的最终扩展度{DC2}。')
    #
    #  # 所选择的前k个种子节点的接近中心性
    # cc = nx.closeness_centrality(G)
    # # closeness_centrality = [cc.get(n) for n in topk_nodes]
    # origin_closeness_centrality = sorted(cc.values(), reverse=True)[:k]
    # origin_closeness_centrality_knodes = sorted(cc, key=cc.get, reverse=True)[:k]
    # logger.info(f'原始接近中心性最高的前{k}个节点分别是{origin_closeness_centrality_knodes}')
    #
    # # CC1 = IC(G, origin_closeness_centrality_knodes, mc)
    # CC2 = IC2(G, origin_closeness_centrality_knodes, mc=1000)
    # # logger.info(f'原始接近中心性最高的前{k}个节点作为种子集的最终传播范围是{CC1}。')
    # logger.info(f'原始接近中心性最高的前{k}个节点作为种子集传播的最终扩展度{CC2}。')
    #
    # # 所选择的前k个种子节点的特征向量中心性
    # ec = nx.eigenvector_centrality(G,max_iter=1000)
    # # eigenvector_centrality = [ec.get(n) for n in topk_nodes]
    # origin_eigenvector_centrality = sorted(ec.values(), reverse=True)[:k]
    # origin_eigenvector_centrality_knodes = sorted(ec, key=ec.get, reverse=True)[:k]
    # logger.info(f'原始特征向量中心性最高的前{k}个节点分别是{origin_eigenvector_centrality_knodes}')
    #
    # # EC1 = IC(G, origin_eigenvector_centrality_knodes, mc)
    # EC2 = IC2(G, origin_eigenvector_centrality_knodes, mc=1000)
    # # logger.info(f'原始特征向量中心性最高的前{k}个节点作为种子集的最终传播范围是{EC1}。')
    # logger.info(f'原始特征向量中心性最高的前{k}个节点作为种子集传播的最终扩展度{EC2}。')
    #
    # # 所选择的前k个种子节点的PageRank值
    # pr = nx.pagerank(G)
    # # pagerank = [pr.get(n) for n in topk_nodes]
    # origin_pagerank = sorted(pr.values(), reverse=True)[:k]
    # origin_pagerank_knodes = sorted(pr, key=pr.get, reverse=True)[:k]
    # logger.info(f'原始pagerank值最高的前{k}个节点分别是{origin_pagerank_knodes}')
    #
    # # PR1 = IC(G, origin_pagerank_knodes, mc)
    # PR2 = IC2(G, origin_pagerank_knodes, mc=1000)
    # # logger.info(f'原始pagerank值最高的前{k}个节点作为种子集的最终传播范围是{PR1}。')
    # logger.info(f'原始pagerank值最高的前{k}个节点作为种子集传播的最终扩展度{PR2}。')


    # S =['322', '208', '190', '385', '192', '254', '269', '147', '303', '111']
    # PR2 = IC2(G, S, mc=1000)
    # logger.info(f'原始pagerank值最高的前{k}个节点作为种子集传播的最终扩展度{PR2}。')


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
