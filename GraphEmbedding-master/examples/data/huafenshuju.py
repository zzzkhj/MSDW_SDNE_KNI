# 打开原始数据文件
if __name__ == '__main__':
    with open('soc-sign-bitcoinalpha.csv', 'r') as f:
        lines = f.readlines()

    # 提取前两列数据
    new_data = []
    for line in lines:
        columns = line.strip().split(',')
        if len(columns) >= 2:
            new_data.append(columns[:2])

    # 将提取的数据保存到新文件中
    with open('soc-sign-bitcoin-alpha.csv', 'w') as f:
        for row in new_data:
            f.write(' '.join(row) + '\n')
