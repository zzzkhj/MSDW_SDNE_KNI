import random
import time
import json  # 用于保存和加载结果
import networkx as nx
import pandas as pd
import sys
import logging
from multiprocessing import Pool
import statistics

from tqdm import tqdm


def SIR_model_spreadability(G, beta=0.037, gamma=0.01, num_simulations=1000):
    # 存储每个节点的传播能力的平均值
    spreadability = {node: 0 for node in G.nodes()}
    # 遍历图中每个节点作为初始感染节点
    bar = tqdm(total=len(G))
    for initial_infected in G.nodes():
        # total_spread = 0  # 累积传播能力
        # 对每个节点进行1000次（num_simulations次）SIR模型模拟
        with Pool(5) as p:
            res = p.starmap(sir, [[G, initial_infected, num_simulations // 5, beta, gamma] for _ in range(5)])
                # 计算该节点的平均传播能力
        spreadability[initial_infected] = statistics.mean(res)
        bar.set_postfix_str(f'{initial_infected}: {statistics.mean(res)}')
        bar.update(1)
    return spreadability

def sir(G, initial_infected, num_simulations, beta, gamma):
    total_spread = 0  # 累积传播能力
    for _ in range(num_simulations):
        S = set(G.nodes())  # Susceptible
        I = {initial_infected}  # Infected
        R = set()  # Recovered
        S.remove(initial_infected)
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
        total_spread += len(total_infected) / G.number_of_nodes()
        # 记录当前模拟中的传播能力（感染到的节点数）
    return  total_spread / num_simulations

def save_results_to_file(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f)

def load_results_from_file(filename):
    with open(filename, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    start_time = time.time()

    # Load graph
    # G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',
    #                      create_using=nx.Graph(), nodetype=None, data=[('weight', int)])
    #
    # results_filename = 'SIR_spreadability_results.json'
    logger = logging.getLogger(__name__)
    # 创建一个handler
    handler = logging.StreamHandler()
    logger.setLevel(logging.DEBUG)  # 设置日志级别为DEBUG，这样所有级别的日志都会被处理
    # 创建一个格式器，并添加到handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # 给logger添加handler
    logger.addHandler(handler)

    network_name_dict = {'1': 'cora', '2': 'email-univ', '3': 'citeseer', '4': 'socfb-Reed98',
                         '5': 'ia-fb-messages', '6': 'bio-CE-GT', '7': 'ca-GrQc', '8': 'power-US-Grid','9':'soc-hamster','10':'bio-CE-LC'}
    network_name_key = input(
        '请输入选择使用的网络编号:1: cora, 2: email-univ, 3: citeseer, 4: socfb-Reed98,5: ia-fb-messages, 6: bio-CE-GT,7:ca-GrQc,8:power-US-Grid,9:soc-hamster,10:bio-CE-LC')
    if network_name_key not in network_name_dict:
        logger.error('请输入1-10中的数字')
        sys.exit(0)
    network_name = network_name_dict[network_name_key]

    if network_name == 'cora':
        data = pd.read_csv('data/cora.edges', sep=',', header=None)
        data.columns = ['source', 'target', 'flag']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
        logger.info(f'使用{network_name}数据集，图有{G.number_of_nodes()}个节点，{G.number_of_edges()}条边。')
        results_filename = '{cora}_SIR_spreadability_results.json'
        beta = 0.101
    elif network_name == 'email-univ':
        data = pd.read_csv('data/email-univ.edges', sep=' ', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
        logger.info(f'使用{network_name}数据集，图有{G.number_of_nodes()}个节点，{G.number_of_edges()}条边。')
        # G = nx.read_edgelist('data/email-univ.edges', create_using=nx.DiGraph(), nodetype=None,
        #                      data=[('weight', int)])
        results_filename = '{email-univ}_SIR_spreadability_results.json'
        beta =0.1
    elif network_name == 'citeseer':
        data = pd.read_csv('data/citeseer.txt', sep=',', header=None)
        data.columns = ['source', 'target', 'flag']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
        logger.info(f'使用{network_name}数据集，图有{G.number_of_nodes()}个节点，{G.number_of_edges()}条边。')
        results_filename = '{citeseer}_SIR_spreadability_results.json'
        beta =0.3
    elif network_name == 'socfb-Reed98':
        data = pd.read_csv('data/socfb-Reed98.mtx', sep=' ', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
        logger.info(f'使用{network_name}数据集，图有{G.number_of_nodes()}个节点，{G.number_of_edges()}条边。')
        results_filename = '{socfb-Reed98}_SIR_spreadability_results.json'
        beta = 0.016
    elif network_name == 'ia-fb-messages':
        data = pd.read_csv('data/ia-fb-messages.mtx', sep=' ', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
        logger.info(f'使用{network_name}数据集，图有{G.number_of_nodes()}个节点，{G.number_of_edges()}条边。')
        results_filename = '{ia-fb-messages}_SIR_spreadability_results.json'
        beta = 0.04
    elif network_name == 'bio-CE-GT':
        data = pd.read_csv('data/bio-CE-GT.edges', sep=' ', header=None)
        data.columns = ['source', 'target', 'flag']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
        logger.info(f'使用{network_name}数据集，图有{G.number_of_nodes()}个节点，{G.number_of_edges()}条边。')
        results_filename = '{bio-CE-GT}_SIR_spreadability_results.json'
        beta = 0.037
    elif network_name == 'ca-GrQc':
        data = pd.read_csv('data/ca-GrQc.mtx', sep=' ', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
        logger.info(f'使用{network_name}数据集，图有{G.number_of_nodes()}个节点，{G.number_of_edges()}条边。')
        results_filename = '{ca-GrQc}_SIR_spreadability_results.json'
        beta =0.05
    elif network_name == 'power-US-Grid':
        data = pd.read_csv('data/power-US-Grid.mtx', sep=' ', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
        logger.info(f'使用{network_name}数据集，图有{G.number_of_nodes()}个节点，{G.number_of_edges()}条边。')
        results_filename = '{power-US-Grid}_SIR_spreadability_results.json'
        beta =0.3
    elif network_name == 'soc-hamster':
        data = pd.read_csv('data/soc-hamsterster.edges', sep=' ', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
        logger.info(f'使用{network_name}数据集，图有{G.number_of_nodes()}个节点，{G.number_of_edges()}条边。')
        results_filename = '{soc-hamster}_SIR_spreadability_results.json'
        beta = 0.03
    elif network_name == 'bio-CE-LC':
        data = pd.read_csv('data/bio-CE-LC.edges', sep=' ', header=None)
        data.columns = ['source', 'target', 'flag']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
        logger.info(f'使用{network_name}数据集，图有{G.number_of_nodes()}个节点，{G.number_of_edges()}条边。')
        results_filename = '{bio-CE-LC}_SIR_spreadability_results.json'
        beta = 0.082
    # 检查结果文件是否存在
    try:
        # 尝试从文件加载结果
        result = load_results_from_file(results_filename)
        print("Loaded results from file.")
    except FileNotFoundError:
        # 如果文件不存在，运行 SIR 模型并保存结果
        beta = beta
        result = SIR_model_spreadability(G, beta)
        save_results_to_file(result, results_filename)
        print("Ran SIR model and saved results to file.")

    # 对节点按传播能力排序
    sorted_spreadability = sorted(result.items(), key=lambda x: x[1], reverse=True)

    # 输出排名结果
    for rank, (node, spread) in enumerate(sorted_spreadability, start=1):
        print(f"Rank {rank}: Node {node} with spreadability {spread:.4f}")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


