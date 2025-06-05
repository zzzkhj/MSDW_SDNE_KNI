import networkx as nx
from ge import SDNE
from sklearn.cluster import KMeans
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, Model

# 配额采样函数
def quota_samples(cluster, num_samples):
    """ 从一个簇中按配额采样 `num_samples` 个节点 """
    return random.sample(list(cluster), min(len(cluster), num_samples))

# 简化版 SIR 扩散模型
def SIR(seed_nodes, threshold):
    """ SIR 扩散模型（简化版），根据种子节点和阈值返回节点的活力值 """
    vitality = {node: random.uniform(0, 1) for node in seed_nodes}
    return vitality

# 自定义的 GAT 模型，使用 TensorFlow 实现
class GATLayer(layers.Layer):
    def __init__(self, output_dim, num_heads=1):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.W = layers.Dense(output_dim)
        self.attention_kernel = layers.Dense(1)

    def call(self, node_features, adj_matrix):
        # 计算线性变换后的节点特征
        node_features_transformed = self.W(node_features)

        # 计算注意力系数
        scores = self.attention_kernel(node_features_transformed)
        attention_coefficients = tf.nn.softmax(scores, axis=1)

        # 使用注意力系数对邻居节点的特征进行加权
        node_features_aggregated = tf.matmul(adj_matrix, node_features_transformed * attention_coefficients)

        return node_features_aggregated

class GATModel(Model):
    def __init__(self, output_dim, num_heads=1):
        super(GATModel, self).__init__()
        self.gat_layer1 = GATLayer(output_dim, num_heads)
        self.gat_layer2 = GATLayer(1, 1)  # 输出维度为 1，用于回归任务

    def call(self, inputs):
        node_features, adj_matrix = inputs
        x = self.gat_layer1(node_features, adj_matrix)
        x = self.gat_layer2(x, adj_matrix)
        return x

if __name__ == "__main__":
    r = 0.01  # 采样比例
    k = 5  # 种子节点数
    b = 0.5  # 扩散阈值

    # 加载图数据
    G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

    # 训练 SDNE 模型
    model = SDNE(G, hidden_size=[256, 128])
    model.train(batch_size=3000, epochs=40, verbose=2)
    embeddings = model.get_embeddings()

    # 将嵌入转换为 NumPy 数组
    embedding_values = np.array(list(embeddings.values()))
    node_list = list(embeddings.keys())

    # 计算需要采样的节点数
    S_num = int(len(G.nodes()) * r)

    # 使用 KMeans 进行聚类
    kmeans = KMeans(n_clusters=k, random_state=0).fit(embedding_values)
    C_list = [np.where(kmeans.labels_ == i)[0] for i in range(k)]

    # 从每个簇中采样
    S_nodes = []
    for c in C_list:
        S_nodes.extend(quota_samples(c, S_num))



    # 获取种子节点的活力值
    Vitality = SIR(S_nodes, b)

    # 打印调试信息
    print("S_nodes:", S_nodes)
    print("node_list:", node_list)
    print("Vitality keys:", list(Vitality.keys()))
    # 准备 GAT 模型训练数据
    try:
        X_train = np.array([embeddings[node_list[node]] for node in S_nodes])
        # y_train = np.array([Vitality[node_list[node]] for node in S_nodes])
        y_train = np.array([Vitality[node] for node in S_nodes if node in Vitality])

    except KeyError as e:
        print(f"KeyError: {e}. Check the node IDs in S_nodes and their presence in Vitality.")
        raise

    # 构建邻接矩阵
    adj_matrix = nx.adjacency_matrix(G).todense()
    adj_matrix = np.array(adj_matrix)

    # 构建 GAT 模型
    gat_model = GATModel(output_dim=64, num_heads=8)
    gat_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.005),
                      loss='mean_squared_error')

    # 训练 GAT 模型
    gat_model.fit([embedding_values, adj_matrix], y_train, epochs=200, batch_size=32, verbose=2)
    # gat_model.fit([X_train, adj_matrix], y_train, epochs=200, batch_size=2405, verbose=2)

    # # 使用整个 embedding_values 而不是 X_train
    # embedding_values_full = embedding_values
    #
    # # 创建掩码，只有 S_nodes 中的节点参与训练
    # mask = np.zeros(len(embedding_values_full), dtype=bool)
    # mask[[node for node in S_nodes]] = True
    #
    # # 将掩码应用到 y_train 和 embeddings_full 上
    # y_train_full = np.zeros(len(embedding_values_full))
    # y_train_full[mask] = y_train
    #
    # # 使用整个邻接矩阵和 embedding_values 进行训练，但只计算 mask 中为 True 的节点的损失
    # gat_model.fit([embedding_values_full, adj_matrix], y_train_full, epochs=200, batch_size=2405, verbose=2)

    # 使用 GAT 模型预测非种子节点的活力值
    predictions = gat_model.predict([embedding_values, adj_matrix])
    for node in set(range(len(G.nodes()))) - set(S_nodes):
        Vitality[node_list[node]] = predictions[node][0]

    # 输出结果
    print("Vitality of nodes:", Vitality)


    # 将预测值映射回节点
    node_vitality = {node: predictions[i][0] for i, node in enumerate(node_list)}

    # 选择活力值最大的 K 个节点作为种子节点
    top_k_nodes = sorted(node_vitality, key=node_vitality.get, reverse=True)[:k]

    # 输出结果
    print("Top K influential nodes:", top_k_nodes)
