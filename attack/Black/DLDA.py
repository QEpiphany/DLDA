import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random
from scipy.sparse import vstack, csr_matrix, coo_matrix
from util.tool import targetItemSelect


class LightGCN(nn.Module):
    def __init__(self, user_num, item_num, embedding_dim=64, n_layers=3, gamma=1.0):
        super(LightGCN, self).__init__()
        self.user_embedding = nn.Embedding(user_num, embedding_dim)
        self.item_embedding = nn.Embedding(item_num, embedding_dim)
        self.n_layers = n_layers
        self.gamma = gamma
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, norm_adj):
        all_user_emb = self.user_embedding.weight
        all_item_emb = self.item_embedding.weight
        emb = torch.cat([all_user_emb, all_item_emb], dim=0)
        embeddings = [emb]
        for _ in range(self.n_layers):
            emb = torch.sparse.mm(norm_adj, emb)
            embeddings.append(emb)
        final_emb = torch.stack(embeddings, dim=1).mean(dim=1)
        user_final, item_final = torch.split(final_emb,
                                             [self.user_embedding.num_embeddings, self.item_embedding.num_embeddings])
        return user_final, item_final

    def get_user_embedding(self, user_idx):
        return self.user_embedding(user_idx)

    def get_item_embedding(self):
        return self.item_embedding.weight

    @staticmethod
    def alignment(x, y, alpha=2):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    @staticmethod
    def uniformity(x, t=2):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def calculate_loss(self, user_e, item_e):
        align = self.alignment(user_e, item_e)
        uniform = self.gamma * (self.uniformity(user_e) + self.uniformity(item_e)) / 2
        return align + uniform


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        # query: [B, 1, D], key/value: [B, N, D] where N is N_templates or target_size
        attn_output, _ = self.attn(query, key, value)  # Output: [B, 1, D]
        return self.proj(attn_output)  # [B, 1, D]


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.block(x)


class LatentUNetDenoiser(nn.Module):
    def __init__(self, dim, time_embed_dim, cross_attn_user, cross_attn_item):
        super().__init__()
        self.res1 = ResBlock(dim)
        self.res2 = ResBlock(dim)
        self.down = nn.Linear(dim, dim)
        self.mid = ResBlock(dim)
        self.cross_attn_user = cross_attn_user
        self.cross_attn_item = cross_attn_item
        self.up = nn.Linear(dim, dim)
        self.res3 = ResBlock(dim)
        self.res4 = ResBlock(dim)
        self.final = nn.Linear(dim + time_embed_dim, dim)

    def forward(self, z, t_embed, cond_user, cond_item):
        # z: [B, D], t_embed: [B, time_embed_dim], cond_user: [N_templates, D], cond_item: [N_targets, D]
        h1 = self.res1(z)  # [B, D]
        h2 = self.res2(h1)  # [B, D]
        d = self.down(h2)  # [B, D]

        # Prepare query
        query = d.unsqueeze(1)  # [B, 1, D]
        batch_size = z.shape[0]
        key_user = cond_user.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N_templates, D]
        key_item = cond_item.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N_targets, D]

        # Cross-attention
        attn_user = self.cross_attn_user(query, key_user, key_user).squeeze(1)  # [B, D]
        attn_item = self.cross_attn_item(query, key_item, key_item).squeeze(1)  # [B, D]

        d = d + attn_user + attn_item  # [B, D]
        m = self.mid(d)  # [B, D]
        u = self.up(m)  # [B, D]
        u = self.res3(u + h2)  # [B, D]
        u = self.res4(u + h1)  # [B, D]

        out = torch.cat([u, t_embed], dim=-1)  # [B, D + time_embed_dim]
        final_out = self.final(out)  # [B, D]
        return final_out, m  # Return final output and intermediate representation


class LatentDiffusionGenerator(nn.Module):
    def __init__(self, latent_dim, hidden_dim=256, num_layers=3, timesteps=1000, device='cuda'):
        super().__init__()
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        self.device = device

        self.betas = torch.linspace(0.0001, 0.02, timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

        self.t_embed_dim = 64
        self.time_embed = nn.Sequential(
            nn.Linear(self.t_embed_dim, hidden_dim),
            nn.ReLU(),
        )

        self.cross_attn_user = CrossAttention(embed_dim=latent_dim, num_heads=4)
        self.cross_attn_item = CrossAttention(embed_dim=latent_dim, num_heads=4)
        self.denoiser = LatentUNetDenoiser(latent_dim, hidden_dim, self.cross_attn_user, self.cross_attn_item)

    def get_sinusoidal_embeddings(self, t, dim):
        half_dim = dim // 2
        emb = torch.exp(torch.arange(half_dim, device=self.device) * -(math.log(10000.0) / (half_dim - 1)))
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    def forward(self, z, t, con_user_emb, con_item_emb):
        t_embed = self.get_sinusoidal_embeddings(t, self.t_embed_dim)
        t_embed = self.time_embed(t_embed)
        output, mid_repr = self.denoiser(z, t_embed, con_user_emb, con_item_emb)
        return output, mid_repr

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + \
               self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def _extract(self, arr, t, shape):
        res = arr.gather(0, t).view(-1, *(1,) * (len(shape) - 1))
        return res.expand(shape)

    def sample(self, con_user_emb, con_item_emb, batch_size):
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
            z = self.p_sample(z, t_tensor, con_user_emb, con_item_emb)
        return z

    def p_sample(self, z, t, con_user_emb, con_item_emb):
        noise_pred, _ = self.forward(z, t, con_user_emb, con_item_emb)
        alpha_t = self._extract(self.alphas, t, z.shape)
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, z.shape)
        beta_t = self._extract(self.betas, t, z.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, z.shape)

        z = (z - beta_t * noise_pred / sqrt_one_minus_alpha_cumprod_t) / torch.sqrt(alpha_t)
        if t[0] > 0:
            noise = torch.randn_like(z)
            z += torch.sqrt(beta_t) * noise
        return z


class DLDA:
    def __init__(self, arg, data):
        self.interact = data.matrix()
        self.userNum = data.user_num
        self.itemNum = data.item_num
        self.maliciousFeedbackSize = arg.maliciousFeedbackSize
        self.maliciousUserSize = arg.maliciousUserSize
        self.targetSize = arg.targetSize
        self.targetItem = targetItemSelect(data, arg)
        print("Target items:", self.targetItem)
        self.targetItem = [data.item[i.strip()] for i in self.targetItem]

        matrix = self.interact.tocoo()
        rows, cols, data = matrix.row, matrix.col, matrix.data
        indices = np.arange(len(rows))
        np.random.shuffle(indices)
        sample_size = int(len(indices) * 0.25)
        sampled_indices = indices[:sample_size]
        sampled_rows = rows[sampled_indices]
        sampled_cols = cols[sampled_indices]
        sampled_data = data[sampled_indices]
        self.sampled_matrix = coo_matrix((sampled_data, (sampled_rows, sampled_cols)),
                                         shape=matrix.shape).tocsr()
        self.userSet = np.unique(sampled_rows)

        self.fakeUserNum = int(self.userNum * self.maliciousUserSize) if self.maliciousUserSize < 1 else int(
            self.maliciousUserSize)
        self.maliciousFeedbackNum = int(self.interact.sum() / self.userNum) if self.maliciousFeedbackSize == 0 else int(
            self.maliciousFeedbackSize * self.itemNum if self.maliciousFeedbackSize < 1 else self.maliciousFeedbackSize)

        dense_matrix = self.sampled_matrix.toarray()
        target_matrix = dense_matrix[:, self.targetItem]
        target_interaction_count = (target_matrix > 0).sum(axis=1)
        self.cond_users = np.where(target_interaction_count > 0)[0]

        if len(self.cond_users) == 0:
            print("Warning: No users have interacted with the target items in the sampled 25% data.")
            num_fallback = max(self.fakeUserNum, 10)
            user_num = dense_matrix.shape[0]
            self.cond_users = np.random.choice(user_num, size=min(num_fallback, user_num), replace=False)

        self.lightgcn = self.train_lightgcn_model(self.sampled_matrix)
        with torch.no_grad():
            self.cond_templates = self.lightgcn.get_user_embedding(
                torch.tensor(self.cond_users, dtype=torch.long).cuda()).cpu().numpy()

        self.item_emb = self.lightgcn.get_item_embedding().detach()
        self.latent_dim = self.item_emb.shape[1]
        self.G = None

        self.recommenderGradientRequired = False
        self.recommenderModelRequired = False

        user_interaction_counts = np.array((self.sampled_matrix > 0).sum(axis=1)).flatten()
        self.real_lambda = user_interaction_counts.mean()

        item_interaction_count = (dense_matrix > 0).sum(axis=0)
        top_items = np.argsort(-item_interaction_count)[:self.itemNum // 10]
        high_pop_matrix = dense_matrix[:, top_items]
        high_pop_interaction_count = (high_pop_matrix > 0).sum(axis=1)
        self.high_pop_users = np.where(high_pop_interaction_count > 0)[0]
        if len(self.high_pop_users) == 0:
            raise ValueError("No users have interacted with high-popularity items in the sampled 25% data.")

    def build_adj_matrix(self, interaction_matrix):
        user_num, item_num = interaction_matrix.shape
        interaction_matrix = interaction_matrix.tocoo()
        row = interaction_matrix.row
        col = interaction_matrix.col + user_num
        edge_index = np.concatenate([np.stack([row, col], axis=0), np.stack([col, row], axis=0)], axis=1)

        value = np.ones(edge_index.shape[1], dtype=np.float32)
        adj = coo_matrix((value, (edge_index[0], edge_index[1])), shape=(user_num + item_num, user_num + item_num))
        row_sum = np.array(adj.sum(1)).flatten()
        d_inv_sqrt = np.power(row_sum, -0.5, where=row_sum != 0)
        d_mat_inv_sqrt = coo_matrix((d_inv_sqrt, (np.arange(len(d_inv_sqrt)), np.arange(len(d_inv_sqrt)))),
                                    shape=adj.shape)
        norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        norm_adj = norm_adj.tocoo()
        i = torch.LongTensor([norm_adj.row, norm_adj.col])
        v = torch.FloatTensor(norm_adj.data)
        return torch.sparse.FloatTensor(i, v, torch.Size(norm_adj.shape)).cuda()

    def bpr_loss(self, users, pos_items, neg_items, user_emb, item_emb):
        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]
        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)
        loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        reg_loss = (u_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / 2
        return loss, reg_loss

    def generate_bpr_triplets(self, interaction_matrix, num_neg=1):
        user_item = {u: set(interaction_matrix[u].nonzero()[1]) for u in range(interaction_matrix.shape[0])}
        triplets = []
        for u in user_item:
            for i in user_item[u]:
                for _ in range(num_neg):
                    j = random.randint(0, interaction_matrix.shape[1] - 1)
                    while j in user_item[u]:
                        j = random.randint(0, interaction_matrix.shape[1] - 1)
                    triplets.append((u, i, j))
        return triplets

    def split_interaction_matrix(self, matrix, val_ratio=0.1):
        matrix = matrix.tocoo()
        rows, cols, data = matrix.row, matrix.col, matrix.data
        indices = np.arange(len(rows))
        np.random.shuffle(indices)
        val_size = int(len(indices) * val_ratio)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        train_rows = rows[train_indices]
        train_cols = cols[train_indices]
        train_data = data[train_indices]
        val_rows = rows[val_indices]
        val_cols = cols[val_indices]
        val_data = data[val_indices]
        train_matrix = coo_matrix((train_data, (train_rows, train_cols)), shape=matrix.shape)
        val_matrix = coo_matrix((val_data, (val_rows, val_cols)), shape=matrix.shape)
        return train_matrix.tocsr(), val_matrix.tocsr()

    def evaluate_hr(self, model, val_matrix, norm_adj, top_k=50):
        model.eval()
        user_emb, item_emb = model(norm_adj)
        scores = torch.matmul(user_emb, item_emb.T).detach().cpu().numpy()
        val_matrix = val_matrix.tocsr()
        hits = 0
        total = 0

        for user in range(val_matrix.shape[0]):
            true_items = val_matrix[user].indices
            if len(true_items) == 0:
                continue
            user_scores = scores[user]
            top_items = np.argsort(user_scores)[-top_k:]
            if np.intersect1d(top_items, true_items).size > 0:
                hits += 1
            total += 1

        return hits / total if total > 0 else 0.0

    def train_lightgcn_model(self, sampled_matrix):
        print("[INFO] Pretraining LightGCN model with DirectAU ...")
        user_num, item_num = sampled_matrix.shape
        train_matrix, val_matrix = self.split_interaction_matrix(sampled_matrix)
        model = LightGCN(user_num, item_num, embedding_dim=64, n_layers=3, gamma=1.0).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        norm_adj = self.build_adj_matrix(train_matrix)
        patience = 20
        best_hr = 0
        epochs_no_improve = 0
        best_model_state = None
        max_epochs = 100
        batch_size = 1024

        for epoch in range(max_epochs):
            model.train()
            triplets = self.generate_bpr_triplets(train_matrix)
            random.shuffle(triplets)
            total_train_loss = 0
            for i in range(0, len(triplets), batch_size):
                batch = triplets[i:i + batch_size]
                users, pos_items, neg_items = zip(*batch)
                users = torch.LongTensor(users).cuda()
                pos_items = torch.LongTensor(pos_items).cuda()
                neg_items = torch.LongTensor(neg_items).cuda()
                user_emb, item_emb = model(norm_adj)
                u_emb = user_emb[users]
                pos_emb = item_emb[pos_items]
                neg_emb = item_emb[neg_items]
                pos_scores = torch.sum(u_emb * pos_emb, dim=1)
                neg_scores = torch.sum(u_emb * neg_emb, dim=1)
                bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
                reg_loss = (u_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / 2
                au_loss = model.calculate_loss(u_emb, pos_emb)
                total = bpr_loss + 1e-4 * reg_loss + 0.1 * au_loss
                optimizer.zero_grad()
                total.backward()
                optimizer.step()
                total_train_loss += total.item()

            avg_train_loss = total_train_loss / (len(triplets) // batch_size + 1)
            model.eval()
            current_hr = self.evaluate_hr(model, val_matrix, norm_adj, top_k=50)
            print(f"[LightGCN Epoch {epoch}] Train Loss: {avg_train_loss:.4f}, HR@50: {current_hr:.4f}")

            if current_hr > best_hr:
                best_hr = current_hr
                best_model_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"[INFO] Early stopping triggered after {epoch + 1} epochs with HR@50 = {best_hr:.4f}.")
                model.load_state_dict(best_model_state)
                break

        model.eval()
        return model

    def project(self, mat, max_n=50, lambda_val=10.0, min_score=0.2):
        if not isinstance(mat, torch.Tensor):
            matrix = torch.tensor(mat, dtype=torch.float32).cuda()
        else:
            matrix = mat.to(dtype=torch.float32, device='cuda')

        batch_size, item_num = matrix.shape
        projected_matrix = torch.zeros_like(matrix)

        for i in range(batch_size):
            user_scores = matrix[i]
            sampled_n = np.random.poisson(lam=lambda_val)
            sampled_n = min(sampled_n, max_n, item_num)

            sorted_vals, sorted_idx = torch.sort(user_scores, descending=True)
            valid_idx = sorted_idx[:sampled_n][sorted_vals[:sampled_n] >= min_score]

            projected_matrix[i, valid_idx] = 1.0

        return projected_matrix.cpu().numpy()


    def dispersive_loss(self, z, tau=0.5):
        D = torch.cdist(z, z, p=2) ** 2
        return torch.log(torch.mean(torch.exp(-D / tau)))

    def posionDataAttack(self, recommender=None, epochs=200, batch_size=32, lambda_dispersive=0.5, tau=0.5):
        # Prepare conditioning embeddings
        con_user_emb = torch.tensor(self.cond_templates, dtype=torch.float32).cuda()  # [N_templates, latent_dim]
        con_item_emb = self.item_emb[self.targetItem]  # [N_targets, latent_dim]

        # Normalize embeddings
        con_user_emb = F.normalize(con_user_emb, dim=1)
        con_item_emb = F.normalize(con_item_emb, dim=1)

        self.recommender = recommender
        if self.G is None:
            self.G = LatentDiffusionGenerator(latent_dim=self.latent_dim, device='cuda').cuda()
            optimizer_G = torch.optim.Adam(self.G.parameters(), lr=0.001)

            for epoch in range(epochs):
                self.G.train()
                userSet = random.sample(list(self.high_pop_users), min(self.fakeUserNum, len(self.high_pop_users)))
                total_loss_epoch = 0.0
                diffusion_loss_epoch = 0.0
                dispersive_loss_epoch = 0.0

                for i in range(0, len(userSet), batch_size):
                    batch_users = userSet[i:i + batch_size]
                    batch_size_actual = len(batch_users)
                    user_idx_tensor = torch.tensor(batch_users, dtype=torch.long).cuda()

                    with torch.no_grad():
                        latent = self.lightgcn.get_user_embedding(user_idx_tensor)  # [batch_size_actual, latent_dim]

                    t = torch.randint(0, self.G.timesteps, (batch_size_actual,), device='cuda').long()
                    noise = torch.randn_like(latent)
                    z_noisy = self.G.q_sample(latent, t, noise)  # [batch_size_actual, latent_dim]

                    # Forward pass with intermediate representation
                    noise_pred, mid_repr = self.G(z_noisy, t, con_user_emb, con_item_emb)
                    diffusion_loss = F.mse_loss(noise_pred, noise)
                    disp_loss = self.dispersive_loss(mid_repr, tau=tau)
                    total_loss = diffusion_loss + lambda_dispersive * disp_loss

                    optimizer_G.zero_grad()
                    total_loss.backward()
                    optimizer_G.step()

                    total_loss_epoch += total_loss.item() * batch_size_actual
                    diffusion_loss_epoch += diffusion_loss.item() * batch_size_actual
                    dispersive_loss_epoch += disp_loss.item() * batch_size_actual

                total_loss_avg = total_loss_epoch / len(userSet)
                diffusion_loss_avg = diffusion_loss_epoch / len(userSet)
                dispersive_loss_avg = dispersive_loss_epoch / len(userSet)
                print(
                    f"[Diffusion Epoch {epoch}] Total Loss: {total_loss_avg:.4f}, Diffusion Loss: {diffusion_loss_avg:.4f}, Dispersive Loss: {dispersive_loss_avg:.4f}")

        self.G.eval()
        fake_rat_list = []
        row, col, entries = [], [], []
        self.fakeUser = list(range(self.userNum, self.userNum + self.fakeUserNum))

        with torch.no_grad():
            for i in range(0, self.fakeUserNum, batch_size):
                batch_size_actual = min(batch_size, self.fakeUserNum - i)
                # Generate fake user embeddings for the batch
                z = self.G.sample(con_user_emb, con_item_emb, batch_size_actual)  # [batch_size_actual, latent_dim]
                fake_rat = torch.matmul(z, self.item_emb.T).sigmoid().cpu().numpy()  # [batch_size_actual, itemNum]

                # Project the batch ratings
                fake_rat_projected = self.project(fake_rat, max_n=self.maliciousFeedbackNum, lambda_val=self.real_lambda)  # [batch_size_actual, itemNum]

                # Collect non-zero indices for the batch
                batch_user_indices = np.arange(i, i + batch_size_actual)
                global_user_indices = np.array(self.fakeUser)[batch_user_indices]
                batch_row, batch_col = np.nonzero(fake_rat_projected)
                batch_row = batch_user_indices[batch_row]
                batch_entries = np.ones_like(batch_row, dtype=np.float32)

                row.extend(batch_row.tolist())
                col.extend(batch_col.tolist())
                entries.extend(batch_entries.tolist())

                for j, u in enumerate(global_user_indices):
                    for c in self.targetItem:
                        row.append(j + i)
                        col.append(c)
                        entries.append(1)

                fake_rat_list.append(fake_rat_projected)

        fake_rat = np.concatenate(fake_rat_list, axis=0)
        fakeRat = csr_matrix((entries, (row, col)), shape=(len(self.fakeUser), self.itemNum), dtype=np.float32)
        return vstack([self.interact, fakeRat])