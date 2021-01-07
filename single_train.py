# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import paddle
paddle.enable_static()
import argparse
import time
import os
import math
from multiprocessing import Process
import numpy as np
import paddle.fluid as F
import paddle.fluid.layers as L
from paddle.distributed import fleet
import pgl
from pgl.utils.logger import log
from pgl import data_loader
from paddle.fluid import profiler

from reader import DeepwalkReader
from model import DeepwalkModel
from utils import get_file_list
from utils import build_gen_func
from utils import load_raw_edges_fn

def init_role():
    log.info("get env TRAINING_ROLE:[%s]" % os.getenv("TRAINING_ROLE"))
    fleet.init()


def optimization(base_lr, loss, train_steps, optimizer='sgd'):
    
    if optimizer == 'sgd':
        optimizer = F.optimizer.SGD(learning_rate=base_lr)
    elif optimizer == 'adam':
        optimizer = F.optimizer.Adam(learning_rate=base_lr)
    else:
        raise ValueError

    log.info('learning rate:%f' % (base_lr))
    #create the DistributeTranspiler configure
    #strategy = fleet.DistributedStrategy()
    #strategy.a_sync = True
    #create the distributed optimizer
    #optimizer = fleet.distributed_optimizer(optimizer, strategy)
    optimizer.minimize(loss)

def train_prog(exe, program, loss, node2vec_pyreader, args, train_steps):
    step = 0
    node2vec_pyreader.start()
	
    profiler.start_profiler("All")
    while True:
        try:
            begin_time = time.time()
            loss_val = exe.run(program, fetch_list=[loss])
            log.info("step %s: loss %.5f speed: %.5f s/step" %
                     (step, np.mean(loss_val), time.time() - begin_time))        
            step += 1
        except F.core.EOFException:
            node2vec_pyreader.reset()

        if step % args.steps_per_save == 0 or step == train_steps:
            profiler.stop_profiler("total", "/tmp/profile")	
            model_save_dir = args.save_path
            model_path = os.path.join(model_save_dir, str(step))
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            #fleet.save_persistables(exe, model_path)
            F.io.save_params(
                exe,
                dirname=model_path,
                main_program=program)
        if step == train_steps:
            break

def test(args):
    data = load_raw_edges_fn(args.edge_path, args.undirected)
    edges = data[0]
    weights = data[1]
    node2idx = data[2]
    unm_nodes = len(node2idx)
    edge_feat={}
    edges_feat["weight"] = np.array(weights)
    graph = pgl.graph.Graph(num_nodes, edges, edge_feat=edge_feat)
    gen_func = build_gen_func(args, graph)

    start = time.time()
    num = 10
    for idx, _ in enumerate(gen_func()):
        if idx % num == num - 1:
            log.info("%s" % (1.0 * (time.time() - start) / num))
            start = time.time()


def walk(args):
    data = load_raw_edges_fn(args.edge_path, args.undirected)
    edges = data[0]
    weights = data[1]
    node2idx = data[2]
    unm_nodes = len(node2idx)
    edges_feat={}
    edges_feat["weight"] = np.array(weights)
    graph = pgl.graph.Graph(num_nodes, edges, edge_feat=edge_feat)
    num_sample_workers = args.num_sample_workers

    if args.train_files is None or args.train_files == "None":
        log.info("Walking from graph...")
        train_files = [None for _ in range(num_sample_workers)]
    else:
        log.info("Walking from train_data...")
        files = get_file_list(args.train_files)
        train_files = [[] for i in range(num_sample_workers)]
        for idx, f in enumerate(files):
            train_files[idx % num_sample_workers].append(f)

    def walk_to_file(walk_gen, filename, max_num):
        with open(filename, "w") as outf:
            num = 0
            for walks in walk_gen:
                for walk in walks:
                    outf.write("%s\n" % "\t".join([str(i) for i in walk]))
                    num += 1
                    if num % 1000 == 0:
                        log.info("Total: %s, %s walkpath is saved. " %
                                 (max_num, num))
                    if num == max_num:
                        return

    m_args = [(DeepwalkReader(
        graph,
        batch_size=args.batch_size,
        walk_len=args.walk_len,
        win_size=args.win_size,
        neg_num=args.neg_num,
        neg_sample_type=args.neg_sample_type,
        walkpath_files=None,
        train_files=train_files[i]).walk_generator(),
               "%s/%s" % (args.walkpath_files, i),
               args.epoch * args.num_nodes // args.num_sample_workers)
              for i in range(num_sample_workers)]
    ps = []
    for i in range(num_sample_workers):
        p = Process(target=walk_to_file, args=m_args[i])
        p.start()
        ps.append(p)
    for i in range(num_sample_workers):
        ps[i].join()


def train(args):
    import logging
    log.setLevel(logging.DEBUG)
    log.info("start")

    worker_num = args.worker_num
    cpu_num = args.cpu_num
    num_devices = int(os.getenv("CPU_NUM", cpu_num))

    data = load_raw_edges_fn(args.edge_path, args.undirected)
    edges = data[0]
    weights = data[1]
    node2idx = data[2]
    num_nodes = len(node2idx)
    print("LEO num_nodes:",num_nodes, len(edges))
    model = DeepwalkModel(num_nodes, args.hidden_size, args.neg_num,
                          args.is_sparse, args.is_distributed, 1.)
    pyreader = model.pyreader
    loss = model.forward()
    # init fleet
    log.info("init_role")

    train_steps = math.ceil(1. * num_nodes * args.epoch /
                            args.batch_size / num_devices / worker_num)
    log.info("Train step: %s" % train_steps)

    if args.optimizer == "sgd":
        args.lr *= args.batch_size * args.walk_len * args.win_size
    optimization(args.lr, loss, train_steps, args.optimizer)

    log.info("start init worker done")
    place = F.CUDAPlace(0) if args.use_cuda else F.CPUPlace()
    exe = F.Executor(place)
    exe.run(F.default_startup_program())
    log.info("Startup done")


    print("LEO num_nodes:",num_nodes, len(edges))
    edges_feat={}
    edges_feat["weight"] = np.array(weights)
    graph = pgl.graph.Graph(num_nodes, edges, edge_feat=edges_feat)
    # bind gen
    gen_func = build_gen_func(args, graph)

    pyreader.decorate_tensor_provider(gen_func)

    train_prog(exe, F.default_main_program(), loss, pyreader, args, train_steps)
    print("fleet try to stop worker\r\n")

if __name__ == '__main__':

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='Deepwalk')
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="Hidden size of the embedding.")
    parser.add_argument(
        "--lr", type=float, default=0.025, help="Learning rate.")
    parser.add_argument(
        "--neg_num", type=int, default=5, help="Number of negative samples.")
    parser.add_argument(
        "--epoch", type=int, default=1, help="Number of training epoch.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Numbert of walk paths in a batch.")
    parser.add_argument(
        "--walk_len", type=int, default=40, help="Length of a walk path.")
    parser.add_argument(
        "--win_size", type=int, default=5, help="Window size in skip-gram.")
    parser.add_argument(
        "--save_path",
        type=str,
        default="model_path",
        help="Output path for saving model.")
    parser.add_argument(
        "--emb_path",
        type=str,
        default="emb",
        help="Output path for embedding.")
    parser.add_argument(
        "--num_sample_workers",
        type=int,
        default=1,
        help="Number of sampling workers.")
    parser.add_argument(
        "--steps_per_save",
        type=int,
        default=3000,
        help="Steps for model saveing.")
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=10000,
        help="Number of nodes in graph.")
    parser.add_argument(
        "--worker_num",
        type=int,
        default=4,
        help="train numbers.")
    parser.add_argument(
        "--cpu_num",
        type=int,
        default=4,
        help="cpu numbers.")
    parser.add_argument("--edge_path", type=str, default="./graph_data")
    parser.add_argument("--train_files", type=str, default=None)
    parser.add_argument("--walkpath_files", type=str, default=None)
    parser.add_argument("--is_distributed", type=str2bool, default=False)
    parser.add_argument("--is_sparse", type=str2bool, default=True)
    parser.add_argument("--undirected", type=str2bool, default=True)
    parser.add_argument("--warm_start_from_dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument(
        "--use_cuda", action='store_true', help="use_cuda", default=False)
    parser.add_argument(
        "--neg_sample_type",
        type=str,
        default="average",
        choices=["average", "outdegree"])
    parser.add_argument(
        "--mode",
        type=str,
        required=False,
        choices=['train', 'walk'],
        default="train")
    parser.add_argument(
        "--optimizer",
        type=str,
        required=False,
        choices=['adam', 'sgd'],
        default="sgd")
    args = parser.parse_args()
    log.info(args)
    if args.mode == "train":
        train(args)
    elif args.mode == "walk":
        walk(args)
