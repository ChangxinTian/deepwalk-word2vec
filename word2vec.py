from input_data import InputData
from random_walk import random_walks_main
import numpy
from model import SkipGramModel
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys


class Word2Vec:
    def __init__(self,
                 input_file_name,
                 output_file_name,
                 emb_dimension=128,
                 batch_size=50,
                 window_size=10,
                 iteration=1,
                 initial_lr=0.025,
                 min_count=5):
        self.data = InputData(input_file_name, min_count)
        self.output_file_name = output_file_name
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.window_size = window_size
        self.iteration = iteration
        self.initial_lr = initial_lr
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)
        self.use_cuda = torch.cuda.is_available()
        self.optimizer = optim.SGD(self.skip_gram_model.parameters(), lr=self.initial_lr)

    def train(self):
        pair_count = self.data.evaluate_pair_count(self.window_size)
        batch_count = self.iteration * pair_count / self.batch_size
        for i in range(int(batch_count)):
            pos_pairs = self.data.get_batch_pairs(self.batch_size,
                                                  self.window_size)
            neg_v = self.data.get_neg_v_neg_sampling(pos_pairs, 5)
            pos_u = [pair[0] for pair in pos_pairs]
            pos_v = [pair[1] for pair in pos_pairs]

            pos_u = Variable(torch.LongTensor(pos_u))
            pos_v = Variable(torch.LongTensor(pos_v))
            neg_v = Variable(torch.LongTensor(neg_v))

            #梯度优化
            self.optimizer.zero_grad()
            loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
            loss.backward()
            self.optimizer.step()

            # 学习率衰减
            if i * self.batch_size % 100000 == 0:
                print("Loss: %0.6f, lr: %0.6f, %d - %d" %(loss.item(), self.optimizer.param_groups[0]['lr'],i,int(batch_count)))
                lr = self.initial_lr * (1.0 - 1.0 * i / batch_count)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                self.skip_gram_model.find_similar(self.data.id2word)

        self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)


if __name__ == '__main__':
    w2v = Word2Vec(input_file_name='./walks_path.txt', output_file_name='word_embedding.txt')
    w2v.train()
