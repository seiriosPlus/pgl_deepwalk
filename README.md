# 数据格式
参考./graph_data/data.txt

# 修改内容
- cluster_train.py
在原始pgl example中distributed_deepwalk的基础上：
* 修改数据导入, 读取args.edge_path；
```python
    data = load_raw_edges_fn(args.edge_path, args.undirected)
    edges = data[0]
    weights = data[1]
    node2idx = data[2]
    num_nodes = len(node2idx)
```
* 参考distribute_metapath2vec实现2.0分布式

- single_train.py
在cluster_train.py基础上，删除分布式代码，改成单机代码

# 运行方式
## 分布式
分布式启动脚本为start.sh，修改pserver ip地址
启动pserver:
./start.sh ps 0
./start.sh ps 1
启动worker:
./start.sh worker 0
./start.sh worker 1

# 问题
1. 分布式在大数据量的时候回挂掉
2. 单机训练速度太慢
3. 没有保存embedding
~
