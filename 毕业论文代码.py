# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 15:47:49 2025

@author: 15047
"""
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*
import networkx as nx
from node2vec import Node2Vec
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from pylab import mpl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# 支持中文
mpl.rcParams['font.sans-serif'] = ['SimHei']

a = pd.read_csv("C:/Users/15047/Desktop/2025.csv")

b = a.drop([42, 43])
b = b.apply(pd.to_numeric)
b1=b.values

c = b.drop(b.columns[[ 0, 43,44]], axis=1)

c = c.apply(pd.to_numeric)
input = c.values
output =b1[:,44]
dc = input / output

plt.hist(dc.flatten(), bins=50, edgecolor='black')
plt.xlabel('Value')
plt.xticks(rotation=45)
plt.ylabel('Frequency')
plt.title('权重分布图')
plt.grid(True)
plt.show()
def replace_values(matrix, threshold):
    # 创建矩阵的副本
    result = matrix.copy()
    # 将小于阈值的元素替换为0
    result[result < threshold] = 0
    return result


threshold = 0.03
# 调用函数
dc2 = replace_values(dc, threshold)
G = nx.DiGraph(dc2)
D = nx.from_numpy_array(dc2)
G.number_of_edges()
# 常用两个汉字词语列表，可根据需要调整
two_char_words = ["农林牧渔业", "煤炭开采和洗选业", "石油和天然气开采业", "金属矿采选业", "非金属矿及其他矿采选业", "食品制造及烟草加工业", "纺织业", "纺织服装鞋帽皮革羽绒及其制品业", "木材加工及家具制造业", "造纸印刷及文教体育用品制造业", "石油加工、炼焦及核燃料加工业", "化学工业", "非金属矿物制品业", "金属冶炼及压延加工业", "金属制品业", "通用、专用设备制造业", "交通运输设备制造业", "电气机械及器材制造业", "通信设备、计算机及其他电子设备制造业", "仪器仪表及文化、办公用机械制造业", "工艺品及其他制造业", "废品废料", "电力、热力的生产和供应业", "燃气生产和供应业", "水的生产和供应业", "建筑业", "交通运输及仓储业", "邮政业", "信息传输、计算机服务和软件业", "批发和零售业", "住宿和餐饮业", "金融业", "房地产业", "租赁和商务服务业", "研究与试验发展业", "综合技术服务业", "水利、环境和公共设施管理业", "居民服务和其他服务业", "教育", "卫生、社会保障和社会福利业", "文化、体育和娱乐业", "公共管理和社会组织"]

# 循环使用词语作为标签
node_labels = {i: two_char_words[i % len(two_char_words)] for i in range(42)}
nx.set_node_attributes(G, node_labels, 'label')

# 获取节点标签用于绘图
labels = nx.get_node_attributes(G, 'label')

# 删除自环
G.remove_edges_from(nx.selfloop_edges(G))


# 定义参数组合列表
parameter_combinations = [
    (1.0, 1.0, 100, 32, 30, 3),
    (2.0, 0.5, 100, 32, 30, 3),
    (0.5, 2.0, 100, 32, 30, 3),
    (1.0, 1.0, 200, 32, 30, 3),
    (1.0, 1.0, 100, 32, 30, 4),
    (1.0, 1.0, 60, 32, 30, 3),
    (1.0, 1.0, 100, 32, 30, 5)
    # 可以添加更多参数组合
]

# 用于存储结果的列表
results = []

for p, q, walk_length, dimensions, num_walks, k in parameter_combinations:
    print(f"正在测试参数组合：p={p}, q={q}, walk_length={walk_length}, dimensions={dimensions}, num_walks={num_walks}, k={k}")

    # 设置 node2vec 参数
    node2vec = Node2Vec(G,
                                dimensions=dimensions,
                                p=p,
                                q=q,
                                walk_length=walk_length,
                                num_walks=num_walks,
                                workers=4)

    walks = node2vec.walks
    walks = [list(map(str, walk)) for walk in walks]

    model = Word2Vec(window=3,
                     min_count=1,
                     batch_words=4)
    model.build_vocab(corpus_iterable=walks)
    model.train(corpus_iterable=walks, total_examples=len(walks), epochs=5)

    # 提取节点嵌入向量
    node_embeddings = []
    for node in G.nodes():
        if node in model.wv:
            node_embeddings.append(model.wv[node])
    node_embeddings = np.array(node_embeddings)

    # 使用KMeans进行聚类，使用当前的k值
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(node_embeddings)

    # 计算轮廓系数
    silhouette_avg = silhouette_score(node_embeddings, cluster_labels)

    results.append((p, q, walk_length, dimensions, num_walks, k, silhouette_avg))

# 打印结果
result_df = pd.DataFrame(results, columns=["p", "q", "walk_length", "dimensions", "num_walks", "k", "silhouette_score"])
print(result_df)







# 设置node2vec参数
node2vec = Node2Vec(G,
                    dimensions=32,  # 嵌入维度
                    p=2,  # 回家参数
                    q=0.5,  # 外出参数
                    walk_length=100,  # 随机游走最大长度
                    num_walks=600,  # 每个节点作为起始节点生成的随机游走个数
                    workers=40  # 并行线程数
                    )

# 训练Node2Vec，参数文档见 gensim.models.Word2Vec
model = node2vec.fit(window=3,  # Skip-Gram窗口大小
                     min_count=1,  # 忽略出现次数低于此阈值的节点（词）
                     batch_words=4  # 每个线程处理的数据量
                     )

# 确保节点顺序一致
node_order = sorted(G.nodes())
X = np.array([model.wv[str(node)] for node in node_order])




degree_sequence = [d for n, d in G.degree()]

plt.hist(degree_sequence, bins = 30)
plt.title("网络度分布图")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.show()
weights = [d['weight'] for u, v, d in G.edges(data = True)]
plt.hist(weights, bins = 30)
plt.title("网络边权重分布图")
plt.xlabel("Edge Weight")
plt.ylabel("Frequency")
plt.show()
# KMeans聚类
cluster_labels = KMeans(n_clusters=3).fit(X).labels_
print(cluster_labels)

# 列出节点标签以及对应的KMeans分类结果
result_df = pd.DataFrame({
    '节点标签': [labels[i] for i in node_order],
    'KMeans分类结果': cluster_labels
})
print(result_df)



# 按节点顺序获取颜色
colors = cluster_labels

plt.figure(figsize=(30, 20))
pos = nx.spring_layout(G, k=0.3)
nx.draw(G, pos, node_color=colors, with_labels=False, node_size=400, font_size=200, arrowsize=20, edge_color='red',
        connectionstyle='arc3,rad=0.2')
nx.draw_networkx_labels(G, pos, labels=labels, font_size=18, font_color='red')
plt.show()

# 基本的网络分析
def network_analysis(G):
    # 计算平均度
    degrees = dict(G.degree())
    average_degree = sum(degrees.values()) / len(degrees)
    # 计算平均聚类系数
    average_clustering = nx.average_clustering(G)
    # 计算平均最短路径长度
    try:
        average_shortest_path = nx.average_shortest_path_length(G)
    except nx.NetworkXError:
        average_shortest_path = "未计算"
    return average_degree, average_clustering, average_shortest_path


# 可视化基本网络分析结果
def visualize_network_metrics(metrics, values, title):
    plt.figure(figsize=(8, 6))
    # 调整柱子宽度为 0.4
    bars = plt.bar(metrics, values, color=['blue', 'green', 'orange'], width=0.4)


    # 调整坐标轴字体大小
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # 调整标题和标签字体大小
    plt.title(title, fontsize=14)
    plt.xlabel('网络指标', fontsize=12)
    plt.ylabel('值', fontsize=12)

    plt.savefig(title.replace(" ", "_") + ".png", dpi=1200)
    plt.rcParams['figure.dpi'] = 300
    plt.show()


# 提取更丰富的节点特征
def extract_node_features(G):
    degrees = dict(G.degree())
    # 度中心性
    degree_centrality = nx.degree_centrality(G)
    # 介数中心性
    betweenness_centrality = nx.betweenness_centrality(G)
    # 接近中心性
    closeness_centrality = nx.closeness_centrality(G)

    node_features = []
    for i in sorted(G.nodes()):
        features = [degrees[i], degree_centrality[i], betweenness_centrality[i], closeness_centrality[i]]
        node_features.append(features)

    return np.array(node_features)


# t - SNE 降维，调整参数
def tsne_reduction(node_features):
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    return tsne.fit_transform(node_features)


# 可视化 t - SNE 降维结果
def visualize_tsne(node_features_2d, labels, title):
    plt.figure(figsize=(8, 8))
    # 生成随机颜色映射
    colors = np.random.rand(len(node_features_2d))
    sc = plt.scatter(node_features_2d[:, 0], node_features_2d[:, 1], c=colors, cmap='viridis')
    for i, label in enumerate(labels):
        plt.annotate(label, (node_features_2d[i, 0], node_features_2d[i, 1]), textcoords="offset points",
                     xytext=(0, 5), ha='center', fontsize=8)
    plt.title(title)
    plt.xlabel('t - SNE 第一维')
    plt.ylabel('t - SNE 第二维')
    # 添加颜色条
    plt.colorbar(sc)
    plt.savefig(title.replace(" ", "_") + ".png", dpi=1200)
    plt.rcParams['figure.dpi'] = 300
    plt.show()


# 模拟节点/边失效
def simulate_failure(G, failure_type, num_failure):
    G_failure = G.copy()
    if failure_type == 'node':
        nodes_to_remove = random.sample(list(G_failure.nodes()), num_failure)
        G_failure.remove_nodes_from(nodes_to_remove)
    elif failure_type == 'edge':
        edges_to_remove = random.sample(list(G_failure.edges()), num_failure)
        G_failure.remove_edges_from(edges_to_remove)
    return G_failure


# 评估网络恢复能力与鲁棒性
def evaluate_resilience(G, failure_type, num_failure, recovery_attempts):
    original_metrics = network_analysis(G)
    resilience_scores = []
    for _ in range(recovery_attempts):
        G_failure = simulate_failure(G, failure_type, num_failure)
        failure_metrics = network_analysis(G_failure)
        score = []
        for i in range(len(original_metrics)):
            if isinstance(original_metrics[i], str) or isinstance(failure_metrics[i], str):
                score.append(0)
            else:
                score.append(1 - abs((failure_metrics[i] - original_metrics[i]) / original_metrics[i]))
        resilience_scores.append(np.mean(score))
    return np.mean(resilience_scores)


# 关键节点与路径分析
def identify_critical_nodes_and_paths(G):
    # 结合介数中心性和特征向量中心性
    betweenness = nx.betweenness_centrality(G)
    eigenvector = nx.eigenvector_centrality(G)
    combined_centrality = {node: betweenness[node] * eigenvector[node] for node in G.nodes()}
    sorted_nodes = sorted(combined_centrality.items(), key=lambda item: item[1], reverse=True)
    critical_nodes = [node for node, _ in sorted_nodes[:10]]

    # 隐性路径分析
    hidden_paths = []
    for source in critical_nodes:
        for target in critical_nodes:
            if source != target:
                try:
                    all_paths = list(nx.all_simple_paths(G, source, target, cutoff=3))
                    if len(all_paths) > 1:
                        for path in all_paths[1:]:
                            hidden_paths.append(path)
                except nx.NetworkXNoPath:
                    continue

    return critical_nodes, hidden_paths


# 对隐性路径按重要性排序
def sort_hidden_paths_by_importance(G, hidden_paths):
    # 计算图中所有节点的介数中心性
    betweenness = nx.betweenness_centrality(G)
    path_importance = []
    for path in hidden_paths:
        # 计算路径上所有节点的介数中心性之和
        node_betweenness_sum = sum([betweenness[node] for node in path])
        # 初始化边权重之和为 0
        edge_weight_sum = 0
        # 遍历路径中的每一对相邻节点
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            # 累加边的权重，如果边不存在则默认权重为 1
            edge_weight_sum += G.get_edge_data(u, v, default={'weight': 1}).get('weight', 1)
        # 计算路径的重要性，即节点介数中心性之和乘以边权重之和
        importance = node_betweenness_sum * edge_weight_sum
        path_importance.append((path, importance))
    # 根据路径的重要性对路径进行降序排序
    sorted_paths = sorted(path_importance, key=lambda x: x[1], reverse=True)
    # 仅返回排序后的路径，不包含重要性数值
    return [path for path, _ in sorted_paths]


# 优化策略模拟
def optimize_network(G, critical_nodes):
    G_optimized = G.copy()
    # 增加关键节点的权重
    for node in critical_nodes:
        for neighbor in G_optimized.neighbors(node):
            if G_optimized.has_edge(node, neighbor):
                G_optimized[node][neighbor]['weight'] = 2

    # 增加冗余路径
    for i in range(5):
        source = random.choice(list(G_optimized.nodes()))
        target = random.choice(list(G_optimized.nodes()))
        if source != target and not G_optimized.has_edge(source, target):
            G_optimized.add_edge(source, target)

    return G_optimized



# 原始图分析
print("原始图的网络分析：")
original_avg_degree, original_avg_clustering, original_avg_shortest_path = network_analysis(D)
metrics = ['平均度', '平均聚类系数']
values = [original_avg_degree, original_avg_clustering]
if original_avg_shortest_path is not None:
    metrics.append('平均最短路径长度')
    values.append(original_avg_shortest_path)
visualize_network_metrics(metrics, values,'1')

original_node_features = extract_node_features(G)
original_node_features_2d = tsne_reduction(original_node_features)
visualize_tsne(original_node_features_2d, [labels[i] for i in sorted(G.nodes())], "原始图 t - SNE 降维结果")

# 模拟节点/边失效并评估恢复能力与鲁棒性
node_resilience = evaluate_resilience(G, 'node', 5, 100)
edge_resilience = evaluate_resilience(G, 'edge', 10, 100)
print(f"节点失效恢复能力得分: {node_resilience}")
print(f"边失效恢复能力得分: {edge_resilience}")

# 关键节点与路径分析
critical_nodes, hidden_paths = identify_critical_nodes_and_paths(G)
print("关键节点:", [labels[node] for node in critical_nodes])

# 计算隐性路径的数量
hidden_paths_count = len(hidden_paths)
print(f"隐性路径的数量: {hidden_paths_count}")

# 对隐性路径按重要性排序
sorted_hidden_paths = sort_hidden_paths_by_importance(G, hidden_paths)
print("按重要性排序后的隐性路径:", [[labels[node] for node in path] for path in sorted_hidden_paths])

# 优化策略模拟
G_optimized = optimize_network(G, critical_nodes)
print("优化后图的网络分析：")
optimized_avg_degree, optimized_avg_clustering, optimized_avg_shortest_path = network_analysis(G_optimized)
metrics = ['平均度', '平均聚类系数']
values = [optimized_avg_degree, optimized_avg_clustering]
if optimized_avg_shortest_path is not None:
    metrics.append('平均最短路径长度')
    values.append(optimized_avg_shortest_path)
visualize_network_metrics(metrics, values, "优化后图基本网络分析结果")

optimized_node_features = extract_node_features(G_optimized)
optimized_node_features_2d = tsne_reduction(optimized_node_features)
visualize_tsne(optimized_node_features_2d, [labels[i] for i in sorted(G_optimized.nodes())], "优化后图 t - SNE 降维结果")

# 生成表格
tables = []

# 表格1：原始网络基本指标
table1 = pd.DataFrame({
    '指标': ['平均度', '平均聚类系数', '平均最短路径长度'],
    '值': [original_avg_degree, original_avg_clustering, original_avg_shortest_path]
})
tables.append(table1)

# 表格2：原始节点特征
feature_names = ['度', '度中心性', '介数中心性', '接近中心性']
table2 = pd.DataFrame(original_node_features, columns=feature_names)
table2['节点标签'] = [labels[i] for i in sorted(G.nodes())]
tables.append(table2)

# 表格3：原始节点聚类结果
table3 = pd.DataFrame({
    '节点标签': [labels[i] for i in sorted(G.nodes())],
    '聚类标签': cluster_labels
})
tables.append(table3)

# 表格4：原始节点t-SNE降维结果
table4 = pd.DataFrame(original_node_features_2d, columns=['t-SNE第一维', 't-SNE第二维'])
table4['节点标签'] = [labels[i] for i in sorted(G.nodes())]
tables.append(table4)

# 表格5：节点失效恢复能力得分
table5 = pd.DataFrame({
    '失效类型': ['节点失效'],
    '恢复能力得分': [node_resilience]
})
tables.append(table5)

# 表格6：边失效恢复能力得分
table6 = pd.DataFrame({
    '失效类型': ['边失效'],
    '恢复能力得分': [edge_resilience]
})
tables.append(table6)

# 表格7：关键节点
table7 = pd.DataFrame({
    '关键节点': [labels[node] for node in critical_nodes]
})
tables.append(table7)

# 表格8：隐性路径数量
table8 = pd.DataFrame({
    '隐性路径数量': [len(hidden_paths)]
})
tables.append(table8)

# 表格9：按重要性排序后的隐性路径
table9 = pd.DataFrame({
    '隐性路径': [[labels[node] for node in path] for path in sorted_hidden_paths]
})
tables.append(table9)

# 表格10：优化后网络基本指标
table10 = pd.DataFrame({
    '指标': ['平均度', '平均聚类系数', '平均最短路径长度'],
    '值': [optimized_avg_degree, optimized_avg_clustering, optimized_avg_shortest_path]
})
tables.append(table10)









import networkx as nx
import matplotlib.pyplot as plt

# 计算中介中心性
betweenness = nx.betweenness_centrality(G)
# 计算特征向量中心性
eigenvector = nx.eigenvector_centrality(G)
# 计算组合中心性
combined_centrality = {node: betweenness[node] * eigenvector[node] for node in G.nodes()}
# 按组合中心性降序排序
sorted_nodes = sorted(combined_centrality.items(), key=lambda item: item[1], reverse=True)
# 选取前10个关键节点
critical_nodes = [node for node, _ in sorted_nodes[:10]]
# 提取前10个关键节点的组合中心性值
critical_centrality_values = [combined_centrality[node] for node in critical_nodes]

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 定义一个函数来对字符串进行分行
def split_string(s, n):
    return '\n'.join([s[i:i+n] for i in range(0, len(s), n)])

# 绘制柱状图
plt.figure(figsize=(15, 8))  # 调整图形大小
# 使用节点的文字标签作为 x 轴标签，并进行分行处理
labels_with_newlines = [split_string(labels[node], 6) for node in critical_nodes]
# 设置柱状图颜色为红色，添加透明度、边框和柱子宽度
bars = plt.bar(labels_with_newlines, critical_centrality_values, color='pink', alpha=0.7, edgecolor='black', width=0.4)

# 添加数据标签
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.4f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom')

# 添加标题和标签
plt.title('Top 10 in 2025 ', fontsize=16, fontweight='bold')
plt.xlabel("行业", fontsize=14)
plt.ylabel('指标', fontsize=14)

# 设置 x 轴标签字体大小和旋转角度
plt.xticks(rotation=30, fontsize=15)
# 设置 y 轴标签字体大小
plt.yticks(fontsize=12)

# 添加网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 显示图形
plt.tight_layout()
plt.show()
    



# 只增加关键节点的权重
def increase_node_weights(G, critical_nodes):
    G_optimized = G.copy()
    for node in critical_nodes:
        for neighbor in G_optimized.neighbors(node):
            if G_optimized.has_edge(node, neighbor):
                G_optimized[node][neighbor]['weight'] = 15
    return G_optimized


# 只增加冗余边
def add_redundant_edges(G, num_edges):
    G_optimized = G.copy()
    for i in range(num_edges):
        source = random.choice(list(G_optimized.nodes()))
        target = random.choice(list(G_optimized.nodes()))
        if source != target and not G_optimized.has_edge(source, target):
            G_optimized.add_edge(source, target)
    return G_optimized

# 只增加关键节点的权重
G_weight_optimized = increase_node_weights(G, critical_nodes)
# 只增加冗余边
G_edge_optimized = add_redundant_edges(G, 10)

original_node_resilience = evaluate_resilience(G, 'node', 4, 100)
original_edge_resilience = evaluate_resilience(G, 'edge', 8, 100)
weight_optimized_node_resilience = evaluate_resilience(G_weight_optimized, 'node', 4, 100)
weight_optimized_edge_resilience = evaluate_resilience(G_weight_optimized, 'edge', 8, 100)
edge_optimized_node_resilience = evaluate_resilience(G_edge_optimized, 'node', 4, 100)
edge_optimized_edge_resilience = evaluate_resilience(G_edge_optimized, 'edge', 8, 100)

print(f"原始图节点失效恢复能力得分: {original_node_resilience}")
print(f"原始图边失效恢复能力得分: {original_edge_resilience}")
print(f"增加节点权重后节点失效恢复能力得分: {weight_optimized_node_resilience}")
print(f"增加节点权重后边失效恢复能力得分: {weight_optimized_edge_resilience}")
print(f"增加冗余边后节点失效恢复能力得分: {edge_optimized_node_resilience}")
print(f"增加冗余边后边失效恢复能力得分: {edge_optimized_edge_resilience}")




