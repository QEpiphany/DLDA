import numpy as np
import torch
import torch.nn as nn
import random
import scipy.sparse as sp
from scipy.sparse import vstack, csr_matrix
from util.tool import targetItemSelect


class AUSH:
    def __init__(self, arg, data):
        self.interact = data.matrix().tocsr()
        self.userNum = data.user_num
        self.itemNum = data.item_num
        self.targetItem = [data.item[i.strip()] for i in targetItemSelect(data, arg)]
        print("Target items:", self.targetItem)

        self.profile_size = arg.profileSize if hasattr(arg, 'profileSize') else 50
        self.selectSize = arg.selectSize if hasattr(arg, 'selectSize') else 5
        self.fakeUserNum = int(self.userNum * arg.maliciousUserSize) if arg.maliciousUserSize < 1 else int(arg.maliciousUserSize)
        self.templateUsers = [u for u in range(self.userNum) if self.interact[u].nnz >= self.profile_size]

        self.G = Generator(self.itemNum, self.selectSize).cuda()
        self.D = Discriminator(self.selectSize).cuda()
        self.itemP = np.array((self.interact.sum(0) / self.interact.sum()))[0]
        self.itemP[self.targetItem] = 0
        self.maliciousFeedbackSize = arg.maliciousFeedbackSize
        self.maliciousUserSize = arg.maliciousUserSize
        if self.maliciousFeedbackSize == 0:
            self.maliciousFeedbackNum = int(self.interact.sum() / data.user_num)
        elif self.maliciousFeedbackSize >= 1:
            self.maliciousFeedbackNum = self.maliciousFeedbackSize
        else:
            self.maliciousFeedbackNum = int(self.maliciousFeedbackSize * self.itemNum)
        if self.maliciousUserSize < 1:
            self.fakeUserNum = int(self.userNum * self.maliciousUserSize)
        else:
            self.fakeUserNum = int(self.maliciousUserSize)
        self.attackForm = "dataAttack"
        self.recommenderGradientRequired = False
        self.recommenderModelRequired = False
        self.BiLevelOptimizationEpoch = 50

    def posionDataAttack(self, epochs=100, batch_size=32, lr=1e-3, lambda_recon=1.0, lambda_shill=1.0):
        G, D = self.G, self.D
        opt_G = torch.optim.Adam(G.parameters(), lr=lr)
        opt_D = torch.optim.Adam(D.parameters(), lr=lr)

        for epoch in range(epochs):
            for _ in range(5):
                D.train()
                users = random.sample(self.templateUsers, batch_size)
                real_vecs, x_ins = [], []
                for u in users:
                    row = self.interact[u].toarray().flatten()
                    fillers = np.random.choice(np.flatnonzero(row), self.profile_size, replace=False)
                    x = np.zeros(self.itemNum)
                    x[fillers] = row[fillers]
                    x_ins.append(x)
                    sel = np.random.choice(list(set(range(self.itemNum)) - set(self.targetItem)), self.selectSize, replace=False)
                    real_vecs.append(row[sel])
                x_ins = torch.FloatTensor(x_ins).cuda()
                real_vecs = torch.FloatTensor(real_vecs).cuda()
                fake_vecs = G(x_ins).detach()
                d_loss = -torch.mean(torch.log(D(real_vecs) + 1e-8) + torch.log(1 - D(fake_vecs) + 1e-8))
                opt_D.zero_grad(); d_loss.backward(); opt_D.step()

            for _ in range(5):
                G.train()
                users = random.sample(self.templateUsers, batch_size)
                x_ins, real_vecs, sel_list = [], [], []
                for u in users:
                    row = self.interact[u].toarray().flatten()
                    fillers = np.random.choice(np.flatnonzero(row), self.profile_size, replace=False)
                    x = np.zeros(self.itemNum)
                    x[fillers] = row[fillers]
                    x_ins.append(x)
                    sel = np.random.choice(list(set(range(self.itemNum)) - set(self.targetItem)), self.selectSize, replace=False)
                    sel_list.append(sel)
                    real_vecs.append(row[sel])
                x_ins = torch.FloatTensor(x_ins).cuda()
                real_vecs = torch.FloatTensor(real_vecs).cuda()
                fake_vecs = G(x_ins)
                adv_loss = -torch.mean(torch.log(D(fake_vecs) + 1e-8))
                recon_loss = nn.MSELoss()(fake_vecs, real_vecs)
                shill_loss = 0
                for i, sel in enumerate(sel_list):
                    idx = [j for j, sid in enumerate(sel) if sid in self.targetItem]
                    if idx:
                        shill_loss += torch.mean((fake_vecs[i][idx] - 1) ** 2)
                shill_loss = shill_loss / batch_size
                g_loss = adv_loss + lambda_recon * recon_loss + lambda_shill * shill_loss
                opt_G.zero_grad(); g_loss.backward(); opt_G.step()

        self.G.eval()
        fake_data = []
        for _ in range(self.fakeUserNum):
            u = random.choice(self.templateUsers)
            row = self.interact[u].toarray().flatten()
            fillers = np.random.choice(np.flatnonzero(row), self.profile_size, replace=False)
            x = np.zeros(self.itemNum)
            x[fillers] = row[fillers]
            x = torch.FloatTensor(x).unsqueeze(0).cuda()
            scores = self.G(x).detach().cpu().numpy().flatten()
            sel = np.random.choice(list(set(range(self.itemNum)) - set(self.targetItem)), self.selectSize, replace=False)
            topk = np.argsort(scores)[-self.selectSize:]
            selected = set([sel[i] for i in topk]) | set(self.targetItem)
            row_idx, col_idx, val = [0]*len(selected), list(selected), [1]*len(selected)
            fake_vec = csr_matrix((val, (row_idx, col_idx)), shape=(1, self.itemNum))
            fake_data.append(fake_vec)
        return vstack([self.interact] + fake_data)


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 400), nn.Sigmoid(),
            nn.Linear(400, 133), nn.Sigmoid(),
            nn.Linear(133, 44), nn.Sigmoid(),
            nn.Linear(44, 14), nn.Sigmoid(),
            nn.Linear(14, output_dim), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)

