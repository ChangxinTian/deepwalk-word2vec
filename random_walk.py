#! /usr/bin/env python
# -*- coding: utf-8 -*-

from io import open
from six.moves import range, zip
import random
from scipy.io import loadmat
from scipy.sparse import issparse


def make_undirected(G):
    for v in G.keys():
        if v in G[v]:
            G[v].remove(v)
        for other in G[v]:
            if v != other:
                if v not in G[v]:
                    G[other].append(v)
    return G


def load_matfile(file_, variable_name="network", undirected=True):
    row_mat_graph = loadmat(file_)
    mat_matrix = row_mat_graph[variable_name]
    G = {}
    if issparse(mat_matrix):
        cx = mat_matrix.tocoo()
        for i, j, v in zip(cx.row, cx.col, cx.data):
            try:
              G[i].append(j)
            except:
              G[i]=[j]
    G = make_undirected(G)
    return G


def get_graph(input_file):
  my_graph={}
  with open(input_file,mode='r',encoding='cp936') as f:
    for i in f.readlines():
      i_list=i.split()
      i_list = list(map(int, i_list))
      my_graph[i_list[0]]=i_list[1:]
  #TODO:必要时要进行undirect
  return my_graph


def random_walk(my_graph,node,walk_length,alpha):
  rand = random.Random()
  walk=[]
  walk.append(node)
  while len(walk) < walk_length:
    cur_node = walk[len(walk)-1]
    if rand.random() >= alpha:
      walk.append(rand.choice(my_graph[cur_node]))
    else:
      walk.append(walk[0])
  return walk


def get_walks(my_graph,walks_per_vectex,walk_length,alpha=0):
  walks = []
  nodes = list(my_graph.keys())
  rand = random.Random()
  for i in range(walks_per_vectex):
    rand.shuffle(nodes)
    for node in nodes:
      cur_walk = random_walk(my_graph,node,walk_length,alpha)
      walks.append(cur_walk)
  return walks


def random_walks_main( input_file="blogcatalog.mat", output_file="walks_path.txt", walks_per_vectex = 80, walk_length=40):

  # print("get graph...")
  # my_graph = get_graph(input_file)
  # print(my_graph)

  print("get graph...")
  my_graph = load_matfile(input_file)
  print(len(my_graph))

  print("get walks...")
  walks = get_walks(my_graph, walks_per_vectex, walk_length)
  walks = [ [str(i) for i in one_walk] for one_walk in walks ]
  fout = open(output_file, 'w', encoding='UTF-8')
  for w in walks:
    one_path = " ".join(w)
    fout.write(one_path+'\n')
  print(len(walks))

if __name__ == "__main__":
  random_walks_main()