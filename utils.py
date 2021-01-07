# -*- coding:utf8 -*-

from __future__ import print_function

import os
import sys
import copy
import glob
import json
import codecs
import ctypes
import random
import pickle
import itertools
import numpy as np
from time import time
from tqdm import tqdm
#from graphsage.graph import NaiveGraph
import multiprocessing as mp
from multiprocessing import Array, Pool, Process, Queue, Lock
from reader import DeepwalkReader
import mp_reader

def get_file_list(path):
    filelist = []
    if os.path.isfile(path):
        filelist = [path]
    elif os.path.isdir(path):
        filelist = [
            os.path.join(dp, f)
            for dp, dn, filenames in os.walk(path) for f in filenames
        ]
    else:
        raise ValueError(path + " not supported")
    return filelist

def build_gen_func(args, graph):
    num_sample_workers = args.num_sample_workers

    if args.walkpath_files is None or args.walkpath_files == "None":
        walkpath_files = [None for _ in range(num_sample_workers)]
    else:
        files = get_file_list(args.walkpath_files)
        walkpath_files = [[] for i in range(num_sample_workers)]
        for idx, f in enumerate(files):
            walkpath_files[idx % num_sample_workers].append(f)

    if args.train_files is None or args.train_files == "None":
        train_files = [None for _ in range(num_sample_workers)]
    else:
        files = get_file_list(args.train_files)
        train_files = [[] for i in range(num_sample_workers)]
        for idx, f in enumerate(files):
            train_files[idx % num_sample_workers].append(f)

    gen_func_pool = [
        DeepwalkReader(
            graph,
            batch_size=args.batch_size,
            walk_len=args.walk_len,
            win_size=args.win_size,
            neg_num=args.neg_num,
            neg_sample_type=args.neg_sample_type,
            walkpath_files=walkpath_files[i],
            train_files=train_files[i]) for i in range(num_sample_workers)
    ]
    if num_sample_workers == 1:
        gen_func = gen_func_pool[0]
    else:
        gen_func = mp_reader.multiprocess_reader(
            gen_func_pool, use_pipe=True, queue_size=100)
    return gen_func

def clean_shared_array(shared_array):
    datastate = shared_array.get_obj()._wrapper._state
    print(datastate)
    arenaobj = datastate[0] #[0]
    arenaobj.buffer.close()
    mp.heap.BufferWrapper._heap = mp.heap.Heap()

def run_random_walks(graph, train_nodes, walk_len=3, num_walks=50):
    tic = time()
    nodes_num = len(graph.nodes)
    nodes = graph.nodes

    neighbors = {}

    # prepare neighbors(nodes must in train set)
    for node in tqdm(train_nodes):
        neighbors[node] = filter(lambda x:x in train_nodes, [np.int64(x.split(":")[0]) for x in graph.adjs[node].split(",")])

    pairs = []
    for count, node in enumerate(tqdm(train_nodes)):
        if len(neighbors[node]) == 0:
            continue

        for i in range(num_walks):
            curr_node = node
            for j in range(walk_len):
                next_node = random.choice(neighbors[curr_node])
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node, curr_node))
                curr_node = next_node
    toc = time()
    print("Random walk done, time cost: {:0.1f} sec.".format(toc-tic))
    return pairs

def label_node(nodes_files_dir, graph, train_ratio=0.7, test_ratio=0.2):
    cache_nodes = os.path.join(nodes_files_dir, "nodes.pkl")
    if os.path.exists(cache_nodes):
        with open(cache_nodes, "rb") as f:
            nodes_dict = pickle.load(f)
        train_nodes = nodes_dict["train"]
        val_nodes = nodes_dict["val"]
        test_nodes = nodes_dict["test"]
    else:
        if not os.path.exists(nodes_files_dir):
            os.mkdir(nodes_files_dir)

        nodes = list(graph.nodes)
        random.shuffle(nodes)
        n_total = len(nodes)
        train_offset = int(n_total * train_ratio)
        test_offset = int(n_total * test_ratio)
        
        # set is implement by r-b tree
        train_nodes = set(nodes[:train_offset])
        test_nodes = set(nodes[train_offset:train_offset+test_offset])
        val_nodes = set(nodes[train_offset+test_offset:])

        nodes_dict = {}
        nodes_dict["train"] = train_nodes
        nodes_dict["val"] = val_nodes
        nodes_dict["test"] = test_nodes
        with open(cache_nodes, "wb") as f:
            pickle.dump(nodes_dict, f)

    return train_nodes, val_nodes, test_nodes

def _calc_lines_num(filename):
    _cmd = "wc -l {}".format(filename)
    with os.popen(_cmd) as _pipe:
        _tmp = _pipe.readline()
    _len = int(_tmp.split(" ")[0])  # line_num filename
    return _len

def _build_features_worker(in_queue, out_queue, lock, feat_shr, feat_shape):
    feat_np = np.frombuffer(feat_shr.get_obj(), dtype=np.float32).reshape(feat_shape)
    while True:
        filename, offset = in_queue.get()
        if filename == "EOF":
            break
        tic = time()
        print('{} start!'.format(filename))
        # do some time cost tasks
        part_id_map = {}
        with open(filename, "r") as f:
            for idx, line in enumerate(f):
                info = line.strip().split(",")
                part_id_map[info[0]] = idx+offset
                feat_np[offset+idx] = info[1:1+feat_shape[1]]
        # put result to out_queue
        with lock:
            out_queue.put(part_id_map)

        print('{} finish, time cost:{:0.1f} sec'.format(filename, time()-tic))

def load_features_fn(feats_files_dir, normalize_label=True, num_workers=48):
    tic = time()
    cache_feats_path = os.path.join(feats_files_dir, 'feats_data.npy')
    cache_node2idx_path = os.path.join(feats_files_dir, 'node2idx.npy')
    cache_idx2node_path = os.path.join(feats_files_dir, 'idx2node.npy')

    if os.path.exists(cache_feats_path) and os.path.exists(cache_node2idx_path) and os.path.exists(cache_idx2node_path): 
        feat_data = np.load(cache_feats_path)
        node2idx = np.load(cache_node2idx_path,allow_pickle=True, encoding="bytes").item()
        idx2node = np.load(cache_idx2node_path,allow_pickle=True, encoding="bytes").item()
    else:
        feat_data = []
        node2idx = {}
        idx2node = {}

        feats_files = sorted(glob.glob("{}/*".format(feats_files_dir)))
        print("feats_files:",feats_files_dir,feats_files)
        with open(feats_files[0], "r") as f:
            tmp_line = f.readline()
            feature_dim = len(tmp_line.strip().split(",")) - 1

        if len(feats_files) == 0:
            raise ValueError("{} is an empty directory".format(feats_files_dir))
        pool = Pool(num_workers)
        files_lens = list(pool.map(_calc_lines_num, feats_files))
        pool.close()
        print("files_lens:", files_lens)
        files_offset = copy.deepcopy(files_lens)
        files_offset.pop(-1)
        files_offset.insert(0, 0)
        for idx in range(1, len(files_lens)):
            files_offset[idx] += files_offset[idx-1]

        print(files_lens)
        print(feature_dim)
        print(files_offset)

        workers = []
        in_queues = []
        out_queue = Queue()
        lock = Lock()
        num_workers = min(num_workers, len(feats_files))
        iters = itertools.cycle(range(num_workers))
        feat_shape = (sum(files_lens), feature_dim)
        feat_data_shr = Array(ctypes.c_float, feat_shape[0]*feat_shape[1])

        for idx in range(num_workers):
            in_queue = Queue()
            in_queues.append(in_queue)
            worker = Process(target=_build_features_worker, args=(in_queue, out_queue, lock, feat_data_shr, feat_shape))
            worker.daemon = True
            worker.start()
            workers.append(worker)

        for idx, filename in enumerate(feats_files):
            queue_idx = next(iters)
            in_queues[queue_idx].put((filename, files_offset[idx]))
        
        for in_queue in in_queues:
            in_queue.put(("EOF", "EOF"))

        for _ in feats_files:
            node2idx.update(out_queue.get())
        
        for key, value in node2idx.items():
            idx2node[value] = key
        
        feat_data = np.frombuffer(feat_data_shr.get_obj(), dtype=np.float32).reshape(feat_shape).copy()
        print(feat_data.shape)

        # FBI warnning; it get error of "cannot close exported pointers exist"
        # I don't know why crrently;
        # clean_shared_array(feat_data_shr)

        print("All Feature Worker Done")

        print("Start feature normalization")
        if normalize_label:
            # 对全图节点进行归一化
            from sklearn.preprocessing import StandardScaler,normalize
            # scaler = StandardScaler()
            # scaler.fit(feat_data)
            # feat_data = scaler.transform(feat_data)
            feat_data = normalize(feat_data)

        # dummy data for adj
        feat_data = np.vstack([feat_data, np.zeros((feat_data.shape[1],))])
        np.save(cache_feats_path, feat_data)
        np.save(cache_node2idx_path, node2idx)
        np.save(cache_idx2node_path, idx2node)

    toc = time()
    print('load features, time cost:{:0.1f} sec'.format(toc-tic))
    return feat_data, node2idx, idx2node

def load_edges_fn(graph_files_dir, node2idx):
    tic = time()
    print("start build graph")
    raw_edges=[]
    weights=[]
    cache_raw_edges=os.path.join(graph_files_dir, "raw_edges.npy")
    cache_weigths=os.path.join(graph_files_dir, "raw_weigths.npy")
    if os.path.exists(cache_raw_edges) and os.path.exists(cache_weigths):
        raw_edges=np.load(cache_raw_edges,allow_pickle=True, encoding="bytes").item()
        weights=np.load(cache_weigths,allow_pickle=True, encoding="bytes")
    else:
        graph_files = sorted(glob.glob("{}/*".format(graph_files_dir)))
        if len(graph_files) == 0:
                raise ValueError("{} is an empty directory".format(graph_files))
        for file in graph_files:
            with open(file, 'r') as f:
                for line in f:
                    line = line.strip().split(",")
                    source = node2idx[line[0]]
                    target = node2idx[line[1]]
                    if line[2] is not None:
                        weights.append(line[2])
                    raw_edges.append((source, target))

        raw_edges = np.array(raw_edges, dtype=np.int64)
        np.save(cache_raw_edges, raw_edges)
        np.save(cache_weigths, weights)

    toc = time()
    print("build graph, time cost:{:0.1f} sec".format(toc-tic))
    return raw_edges,weights

# 有feature情况下加载数据图
def load_data(prefix):
    feature_files_dir = os.path.join(prefix, "feature")
    graph_files_dir = os.path.join(prefix, "graph")
    # load features and id_map
    print("load_features_fn")
    raw_feats, node2idx, idx2node = load_features_fn(feature_files_dir)
    # load graph
    raw_edges,weights = load_edges_fn(graph_files_dir, node2idx)
    return raw_feats, raw_edges, weights, node2idx, idx2node

# 从边的csv文件中加载边，节点idx，weights
def load_raw_edges_fn(graph_files_dir,undirected):
    tic = time()
    print("start build graph")
    raw_edges=[]
    weights=[]
    node2idx = {}
    idx2node = {}

    cache_raw_edges=os.path.join(graph_files_dir, "raw_edges.npy")
    cache_weights=os.path.join(graph_files_dir, "raw_weigths.npy")
    cache_node2idx=os.path.join(graph_files_dir, 'node2idx.npy')
    cache_idx2node=os.path.join(graph_files_dir, 'idx2node.npy')

    if os.path.exists(cache_raw_edges) and os.path.exists(cache_idx2node) and os.path.exists(cache_node2idx) and os.path.exists(cache_weights) :
        raw_edges=np.load(cache_raw_edges)
        weights=np.load(cache_weights,allow_pickle=True, encoding="bytes")
        node2idx=np.load(cache_node2idx,allow_pickle=True, encoding="bytes").item()
        idx2node=np.load(cache_idx2node,allow_pickle=True, encoding="bytes").item()
    else:
        graph_files = sorted(glob.glob("{}/*".format(graph_files_dir)))
        if len(graph_files) == 0:
                raise ValueError("{} is an empty directory".format(graph_files))

        ##ToDo采用multi process方式
        index = 0    
        print("Leo Warnning  : standard data should split by comma")
        for file in graph_files:
            with open(file, 'r') as f:
                for line in f:
                    line = line.strip().split(" ")
                    if len(line) > 3 or len(line) < 2:
                        print("line:",line)
                    assert len(line) in [2, 3], "The format of network file is unrecognizable." 
                    
                    if line[0] not in node2idx:
                        node2idx[line[0]]=index
                        index+=1
                    if line[1] not in node2idx:
                        node2idx[line[1]]=index
                        index+=1
                    
                    source = node2idx[line[0]]
                    target = node2idx[line[1]]

                    if line[2] is not None:
                        weights.append(line[2])
                    raw_edges.append((source, target))

                    #边是否有向
                    if undirected:
                        if line[2] is not None:
                            weights.append(line[2])
                        raw_edges.append((target, source))
        print("after load raw edges")
        raw_edges = np.array(raw_edges, dtype=np.int64)
        np.save(cache_raw_edges, raw_edges)
        np.save(cache_weights, weights)
        np.save(cache_node2idx, node2idx)
        np.save(cache_idx2node, idx2node)

    toc = time()
    print("build graph, time cost:{:0.1f} sec".format(toc-tic))
    return raw_edges,weights,node2idx,idx2node 

# 只有边没有feature情况下加载边信息，
def load_data_without_features(prefix):
    graph_files_dir = os.path.join(prefix, "graph")
    # load graph
    raw_edges,weights,node2idx,idx2node = load_raw_edges_fn(graph_files_dir, True)
    return raw_edges, weights, node2idx, idx2node
