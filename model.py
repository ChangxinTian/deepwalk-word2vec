import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib


class SkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.u_embeddings.weight.data.uniform_(-0.5 / self.emb_dimension, 0.5 / self.emb_dimension)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_v):
        #pos_u[50]
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        # emb_v[50,100]
        score = torch.mul(emb_u, emb_v)
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)
        # neg_v[50,5]
        neg_emb_v = self.v_embeddings(neg_v)
        # neg_emb_v[50,5,100]
        neg_score = torch.bmm(neg_emb_v, emb_u.unsqueeze(2)).squeeze()
        # neg_score[50,5] <- [50,5,100]*[50,100,1]
        neg_score = F.logsigmoid(-1 * neg_score)
        neg_score = torch.sum(neg_score, dim=1)

        final_score = -1 * (torch.sum(score) + torch.sum(neg_score))
        return final_score


    def save_embedding(self, id2word, file_name):
        fout = open(file_name, 'w', encoding='UTF-8')
        fout.write('emb_size: %d \temb_dimension%d\n' % (len(id2word), self.emb_dimension))
        embedding = self.u_embeddings.weight.data.numpy()
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))


    def show_the_embedding(self):
        font = {'family': 'SimHei'}
        matplotlib.rc('font', **font)
        matplotlib.rcParams['axes.unicode_minus'] = False

        viz_words = 33
        tsne = TSNE()
        embed_tsne = tsne.fit_transform(embedding[:viz_words, :])
        for idx in range(viz_words):
            plt.scatter(*embed_tsne[idx, :], color='steelblue')
            plt.annotate(id2word[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
        plt.show()


    def find_similar(self, id2word):
        print('similarity:')
        embedding = self.u_embeddings.weight.data.numpy()
        my_kes = id2word.keys()
        ids = np.random.choice(len(my_kes), 5)
        for id in ids:
            emb1 = embedding[id]
            tagert = {}
            temp_tagert = ()
            for i, emb2 in enumerate(embedding):
                op = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * (np.linalg.norm(emb2)))
                scores = tagert.values()
                if len(scores) < 10:
                    tagert[id2word[i]] = op
                else:
                    tagert[id2word[i]] = op
                    result_mix = min(tagert, key=lambda x: tagert[x])
                    tagert.pop(result_mix)
                    temp_tagert = sorted(tagert.items(), key=lambda x: x[1], reverse=True)
            print(id, ':', id2word[id], temp_tagert)
        # emb1 = embedding[4259]
        # tagert = {}
        # temp_tagert = ()
        # for i, emb2 in enumerate(embedding):
        #     op = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * (np.linalg.norm(emb2)))
        #     scores = tagert.values()
        #     if len(scores) < 10:
        #         tagert[id2word[i]] = op
        #     else:
        #         tagert[id2word[i]] = op
        #         result_mix = min(tagert, key=lambda x: tagert[x])
        #         tagert.pop(result_mix)
        #         temp_tagert = sorted(tagert.items(), key=lambda x: x[1], reverse=True)
        # print(4259, ':', id2word[4259], temp_tagert)