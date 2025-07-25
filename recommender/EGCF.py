import torch
import torch.nn as nn
from util.sampler import next_batch_pairwise
from util.loss import bpr_loss, l2_reg_loss
import time
from util.algorithm import find_k_largest
from time import strftime, localtime, time
from os.path import abspath
import sys
from util.metrics import ranking_evaluation
from util.FileIO import FileIO
from util.logger import Log
import scipy.sparse as sp
import numpy as np


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def info_nce_loss(emb1, emb2, temperature):
    """Simplified InfoNCE loss implementation for self-supervised learning."""
    emb1 = nn.functional.normalize(emb1, dim=1)
    emb2 = nn.functional.normalize(emb2, dim=1)
    similarity = torch.matmul(emb1, emb2.t()) / temperature
    labels = torch.arange(emb1.size(0)).to(emb1.device)
    return nn.CrossEntropyLoss()(similarity, labels)


class EGCF:
    def __init__(self, args, data):
        print("Recommender: EGCF")
        self.data = data
        self.args = args
        self.bestPerformance = []
        self.recOutput = []
        top = self.args.topK.split(',')
        self.topN = [int(num) for num in top]
        self.max_N = max(self.topN)
        # Validate dataset dimensions
        self.learn_rate = 0.001
        self.reg_lambda = 0.0001
        self.GCN_layer = 3
        self.ssl_lambda = 0.1
        self.temperature = 0.1
        self.mode = 'parallel'
        self.model = EGCF_Encoder(self.data, self.args.emb_size, self.GCN_layer, self.mode,
                                  self.reg_lambda, self.ssl_lambda, self.temperature)

    def train(self, requires_grad=False, requires_adjgrad=False, requires_embgrad=False, gradIterationNum=10, Epoch=0,
              optimizer=None, evalNum=5):
        self.bestPerformance = []
        model = self.model.cuda()
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lRate)
        else:
            self.optimizer = optimizer
        if requires_grad or requires_embgrad:
            model.requires_grad = True
            self.itemgrad = torch.zeros((self.data.item_num, self.args.emb_size)).cuda()
        if requires_grad or requires_adjgrad:
            self.model.user_graph.requires_grad = True
            if self.args.mode == 'parallel':
                self.model.graph.requires_grad = True
            self.Matgrad = torch.zeros((self.data.user_num + self.data.item_num,
                                        self.data.user_num + self.data.item_num)).cuda()
        maxEpoch = self.args.maxEpoch
        if Epoch:
            maxEpoch = Epoch
        for epoch in range(maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.args.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                model.train()
                user_emb, item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = user_emb[user_idx], item_emb[pos_idx], item_emb[neg_idx]
                ego_pos_emb, ego_neg_emb = model.embedding_dict['item_emb'][pos_idx], model.embedding_dict['item_emb'][
                    neg_idx]

                # Compute losses
                batch_bpr_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                batch_reg_loss = l2_reg_loss(self.reg_lambda, ego_pos_emb, ego_neg_emb)
                batch_ssl_loss = self.ssl_lambda * (
                        info_nce_loss(user_emb, user_emb, self.temperature) +
                        info_nce_loss(pos_item_emb, pos_item_emb, self.temperature) +
                        info_nce_loss(user_emb, pos_item_emb, self.temperature)
                )
                batch_loss = batch_bpr_loss + batch_reg_loss + batch_ssl_loss

                self.optimizer.zero_grad()
                batch_loss.backward()

                if (requires_grad or requires_adjgrad) and maxEpoch - epoch < gradIterationNum:
                    self.Matgrad += self.model.user_graph.grad
                    if self.args.mode == 'parallel':
                        self.Matgrad += self.model.graph.grad
                if (requires_grad or requires_embgrad) and maxEpoch - epoch < gradIterationNum:
                    self.itemgrad += self.model.embedding_dict['item_emb'].grad

                self.optimizer.step()
                if n % 1000 == 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            model.eval()
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            if epoch % evalNum == 0:
                self.evaluate(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        if (requires_grad or requires_adjgrad) and (requires_grad or requires_embgrad):
            return (self.Matgrad + self.Matgrad.T)[:self.data.user_num, self.data.user_num:], \
                self.user_emb, self.item_emb, self.itemgrad
        elif requires_grad or requires_adjgrad:
            return (self.Matgrad + self.Matgrad.T)[:self.data.user_num, self.data.user_num:]
        elif requires_grad or requires_embgrad:
            return self.user_emb, self.item_emb, self.itemgrad

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            score = torch.sigmoid(score)
            return score.cpu().numpy()

    def evaluate(self, epoch):
        print('Evaluating the model...')
        rec_list, _ = self.test()
        measure = ranking_evaluation(self.data.test_set, rec_list, [self.max_N])
        if len(self.bestPerformance) > 0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            for k in self.bestPerformance[1]:
                if self.bestPerformance[1][k] > performance[k]:
                    count += 1
                else:
                    count -= 1
            if count < 0:
                self.bestPerformance[1] = performance
                self.bestPerformance[0] = epoch + 1
                self.save()
        else:
            self.bestPerformance.append(epoch + 1)
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            self.bestPerformance.append(performance)
            self.save()
        print('-' * 120)
        print('Real-Time Ranking Performance ' + ' (Top-' + str(self.max_N) + ' Item Recommendation)')
        measure = [m.strip() for m in measure[1:]]
        print('*Current Performance*')
        print('Epoch:', str(epoch + 1) + ',', '  |  '.join(measure))
        bp = ''
        bp += 'Hit Ratio' + ':' + str(self.bestPerformance[1]['Hit Ratio']) + '  |  '
        bp += 'Precision' + ':' + str(self.bestPerformance[1]['Precision']) + '  |  '
        bp += 'Recall' + ':' + str(self.bestPerformance[1]['Recall']) + '  |  '
        bp += 'NDCG' + ':' + str(self.bestPerformance[1]['NDCG'])
        print('*Best Performance* ')
        print('Epoch:', str(self.bestPerformance[0]) + ',', bp)
        print('-' * 120)
        return measure

    def test(self):
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum * 2)
            sys.stdout.write(r)
            sys.stdout.flush()

        rec_list = {}
        user_count = len(self.data.test_set)
        for i, user in enumerate(self.data.test_set):
            candidates = self.predict(user)
            rated_list, li = self.data.user_rated(user)
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8
            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')
        return rec_list, ranking_evaluation(self.data.test_set, rec_list, self.topN)


class EGCF_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers, mode, reg_lambda, ssl_lambda, temperature):
        super(EGCF_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.mode = mode
        self.reg_lambda = reg_lambda
        self.ssl_lambda = ssl_lambda
        self.temperature = temperature
        self.user_graph = self._init_user_graph()
        if self.mode == 'parallel':
            self.graph = self._init_full_graph()
        self.embedding_dict = self._init_model()
        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def _init_user_graph(self):
        """Initialize user-item bipartite graph."""
        ui_adj = self.data.norm_adj
        # If norm_adj is a full graph, extract user-item part
        if ui_adj.shape[0] == ui_adj.shape[1] == (self.data.user_num + self.data.item_num):
            ui_adj = ui_adj[:self.data.user_num, self.data.user_num:]
            print(f"Extracted user-item adj from full graph, new shape: {ui_adj.shape}")
        # Validate dimensions
        if ui_adj.shape != (self.data.user_num, self.data.item_num):
            print(
                f"Warning: norm_adj shape {ui_adj.shape} does not match expected ({self.data.user_num}, {self.data.item_num})")
            ui_adj = ui_adj[:, :self.data.item_num]  # Truncate to correct item count
            print(f"Truncated norm_adj to shape: {ui_adj.shape}")
        return trans_to_cuda(TorchGraphInterface.convert_sparse_mat_to_tensor(ui_adj))

    def _init_full_graph(self):
        """Initialize full graph for parallel mode."""
        ui_adj = self.data.norm_adj
        expected_size = self.data.user_num + self.data.item_num
        # If norm_adj is already a full graph, validate size
        if ui_adj.shape == (expected_size, expected_size):
            full_adj = ui_adj
        else:
            # Assume norm_adj is user-item, construct full graph
            if ui_adj.shape != (self.data.user_num, self.data.item_num):
                print(
                    f"Warning: norm_adj shape {ui_adj.shape} does not match expected ({self.data.user_num}, {self.data.item_num})")
                ui_adj = ui_adj[:, :self.data.item_num]
                print(f"Truncated norm_adj to shape: {ui_adj.shape} for full graph")
            full_adj = sp.bmat([[None, ui_adj], [ui_adj.T, None]])
            full_adj = sp.csr_matrix(full_adj)
        if full_adj.shape != (expected_size, expected_size):
            raise ValueError(
                f"Full graph shape {full_adj.shape} does not match expected ({expected_size}, {expected_size})")
        return trans_to_cuda(TorchGraphInterface.convert_sparse_mat_to_tensor(full_adj))

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size)))
        })
        return embedding_dict

    def alternating_aggregate(self):
        item_embedding = self.embedding_dict['item_emb']
        all_user_embeddings = []
        all_item_embeddings = []

        for _ in range(self.layers):
            user_embedding = self.activation(torch.sparse.mm(self.user_graph, item_embedding))
            item_embedding = self.activation(torch.sparse.mm(self.user_graph.t(), user_embedding))
            all_user_embeddings.append(user_embedding)
            all_item_embeddings.append(item_embedding)

        final_user_embeddings = torch.stack(all_user_embeddings, dim=1).sum(dim=1)
        final_item_embeddings = torch.stack(all_item_embeddings, dim=1).sum(dim=1)
        return final_user_embeddings, final_item_embeddings

    def parallel_aggregate(self):
        item_embedding = self.embedding_dict['item_emb']
        user_embedding = self.activation(torch.sparse.mm(self.user_graph, item_embedding))
        all_embedding = torch.cat([user_embedding, item_embedding], dim=0)
        all_embeddings = []

        for _ in range(self.layers):
            all_embedding = self.activation(torch.sparse.mm(self.graph, all_embedding))
            all_embeddings.append(all_embedding)

        final_all_embeddings = torch.stack(all_embeddings, dim=1).sum(dim=1)
        user_emb, item_emb = torch.split(final_all_embeddings, [self.data.user_num, self.data.item_num])
        return user_emb, item_emb

    def forward(self):
        if self.mode == 'parallel':
            return self.parallel_aggregate()
        else:
            return self.alternating_aggregate()


class TorchGraphInterface:
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor(X):
        coo = X.tocoo()
        i = torch.LongTensor(np.array([coo.row, coo.col]))
        v = torch.from_numpy(coo.data).float()
        return torch.sparse_coo_tensor(i, v, coo.shape)