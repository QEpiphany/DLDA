# -*- coding: utf-8 -*-
import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import vstack, csr_matrix, coo_matrix
from util.tool import targetItemSelect



class AE(nn.Module):
    def __init__(self, in_dim, hid=512, zdim=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, zdim))
        self.dec = nn.Sequential(nn.Linear(zdim, hid), nn.ReLU(), nn.Linear(hid, in_dim))
    def encode(self, x):
        return self.enc(x)
    def decode(self, z):
        return self.dec(z)


def build_norm_adj(matrix):
    u, i = matrix.shape
    mat = matrix.tocoo()
    row_u = mat.row
    col_i = mat.col + u
    ei = np.concatenate([np.stack([row_u, col_i], 0),
                         np.stack([col_i, row_u], 0)], 1)
    val = np.ones(ei.shape[1], dtype=np.float32)
    A = coo_matrix((val, (ei[0], ei[1])), shape=(u + i, u + i))
    d = np.array(A.sum(1)).flatten()
    d_inv_sqrt = np.power(d, -0.5, where=d != 0)
    Dinv = coo_matrix((d_inv_sqrt, (np.arange(len(d_inv_sqrt)), np.arange(len(d_inv_sqrt)))), shape=A.shape)
    Ahat = (Dinv @ A @ Dinv).tocsr()
    return Ahat


def initialize_item_from_users(Eu, matrix):
    U, I = matrix.shape
    deg_i = np.array(matrix.sum(0)).flatten() + 1e-6

    if isinstance(Eu, torch.Tensor):
        Eu = Eu.detach().cpu().numpy()

    Ei = matrix.T.dot(Eu) / deg_i[:, None]
    return Ei.astype(np.float32)



def propagate_global(Eu, matrix, K=1):
    if isinstance(Eu, torch.Tensor):
        Eu_np = Eu.detach().cpu().numpy()
    else:
        Eu_np = Eu

    U, I = matrix.shape

    Ei0_np = initialize_item_from_users(Eu_np, matrix)

    X0_np = np.vstack([Eu_np, Ei0_np]).astype(np.float32)


    Ahat = build_norm_adj(matrix).tocoo()
    idx = torch.tensor([Ahat.row, Ahat.col], dtype=torch.long, device='cuda')
    val = torch.tensor(Ahat.data, dtype=torch.float32, device='cuda')
    A_sp = torch.sparse_coo_tensor(idx, val, torch.Size(Ahat.shape))


    X = torch.from_numpy(X0_np).to('cuda')
    for _ in range(K):
        X = torch.sparse.mm(A_sp, X)

    Xu, Xi = X[:U], X[U:]
    return Xu, Xi




class DDPM(nn.Module):
    def __init__(self, zdim, T=1000, beta_start=1e-4, beta_end=2e-2, t_embed_dim=64):
        super().__init__()
        self.zdim = zdim
        self.T = T
        self.register_buffer("betas", torch.linspace(beta_start, beta_end, T))
        self.register_buffer("alphas", 1. - self.betas)
        self.register_buffer("alphas_bar", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("sqrt_ab", torch.sqrt(self.alphas_bar))
        self.register_buffer("sqrt_one_minus_ab", torch.sqrt(1. - self.alphas_bar))
        self.register_buffer("inv_sqrt_alpha", torch.sqrt(1. / self.alphas))
        self.register_buffer("sigma_t", torch.sqrt(self.betas))
        self.t_dim = t_embed_dim
        self.t_mlp = nn.Sequential(nn.Linear(self.t_dim, 256), nn.SiLU(), nn.Linear(256, 256))

    def t_embed(self, t):
        half = self.t_dim // 2
        freq = torch.exp(torch.arange(half, device=t.device) * (-math.log(10000.0) / (half - 1)))
        ang = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        if self.t_dim % 2 == 1:
            emb = torch.cat([emb, emb[:, :1]*0], dim=1)
        return self.t_mlp(emb)

    def q_sample(self, z0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(z0)
        sab = self.sqrt_ab.gather(0, t).view(-1, 1)
        s1m = self.sqrt_one_minus_ab.gather(0, t).view(-1, 1)
        return sab * z0 + s1m * noise, noise

    def p_step(self, z_t, eps_pred, t):
        beta_t = self.betas.gather(0, t).view(-1, 1)
        alpha_t = self.alphas.gather(0, t).view(-1, 1)
        ab_t = self.alphas_bar.gather(0, t).view(-1, 1)
        mean = (1. / torch.sqrt(alpha_t)) * (z_t - (beta_t / torch.sqrt(1 - ab_t)) * eps_pred)
        out = mean
        nonzero = (t > 0).float().view(-1, 1)
        noise = torch.randn_like(z_t)
        out = mean + nonzero * self.sigma_t.gather(0, t).view(-1, 1) * noise
        return out

    def predict_x0(self, z_t, eps_pred, t):
        ab_t = self.alphas_bar.gather(0, t).view(-1, 1)
        s1m = self.sqrt_one_minus_ab.gather(0, t).view(-1, 1)
        return (z_t - s1m * eps_pred) / (torch.sqrt(ab_t) + 1e-8)



class ToA(nn.Module):
    """
    ϵθ(e_{u;s}, s, e_t):
      b_{u;s} = E'(e_{u;s})
      τ(e_t)  = Linear -> 与瓶颈对齐
      Attn(Q=W_Q b, K=W_K τ(e_t), V=W_V τ(e_t))
      h = D'( b + Attn )
      ϵ̂ = W_out(h)
    """
    def __init__(self, zdim, bottleneck=128):
        super().__init__()
        self.enc_p = nn.Sequential(nn.Linear(zdim, bottleneck), nn.GELU())
        self.dec_p = nn.Sequential(nn.Linear(bottleneck, zdim), nn.GELU())
        self.t_proj = nn.Linear(zdim, bottleneck, bias=False)
        self.Wq = nn.Linear(bottleneck, bottleneck, bias=False)
        self.Wk = nn.Linear(bottleneck, bottleneck, bias=False)
        self.Wv = nn.Linear(bottleneck, bottleneck, bias=False)
        self.out = nn.Linear(zdim + 256, zdim)

    def cross_attn(self, b, t_feat):
        Q = self.Wq(b).unsqueeze(1)
        K = self.Wk(t_feat)
        V = self.Wv(t_feat)
        att = torch.softmax((Q * K).sum(-1) / (K.size(-1) ** 0.5), dim=1)
        ctx = (att.unsqueeze(-1) * V).sum(1)
        return ctx

    def forward(self, z_s, t_embed, e_t):
        B = z_s.size(0)
        b = self.enc_p(z_s)
        if e_t.dim() == 2:
            e_t = e_t.unsqueeze(0).expand(B, -1, -1)
        t_feat = self.t_proj(e_t)
        ctx = self.cross_attn(b, t_feat)
        h = self.dec_p(b + ctx)
        eps_hat = self.out(torch.cat([h, t_embed], dim=-1))
        return eps_hat



class ToDA:
    def __init__(self, arg, data):
        self.interact = data.matrix().tocsr()
        self.userNum = data.user_num
        self.itemNum = data.item_num

        tgt_raw = targetItemSelect(data, arg)
        print("Target items:", tgt_raw)
        self.targetItem = [data.item[i.strip()] for i in tgt_raw]


        self.zdim = getattr(arg, "toda_zdim", 64)
        self.hid = getattr(arg, "toda_ae_hidden", 512)
        self.T = getattr(arg, "toda_T", 1000)
        self.beta_start = getattr(arg, "toda_beta_start", 1e-4)
        self.beta_end = getattr(arg, "toda_beta_end", 2e-2)
        self.bottleneck = getattr(arg, "toda_bottleneck", 128)
        self.K_prop = getattr(arg, "toda_Kprop", 1)


        musz = getattr(arg, "maliciousUserSize", 0.05)
        self.fakeUserNum = int(self.userNum * musz) if musz < 1 else int(musz)
        mfs = getattr(arg, "maliciousFeedbackSize", 0)
        if mfs == 0:
            self.maliciousFeedbackNum = int(self.interact.sum() / self.userNum)
        elif mfs < 1:
            self.maliciousFeedbackNum = int(mfs * self.itemNum)
        else:
            self.maliciousFeedbackNum = int(mfs)


        self.ae = AE(self.itemNum, hid=self.hid, zdim=self.zdim).cuda()


        self.diff = DDPM(self.zdim, T=self.T, beta_start=self.beta_start, beta_end=self.beta_end).cuda()
        self.toa = ToA(self.zdim, bottleneck=self.bottleneck).cuda()


        mat = self.interact.tocoo()
        deg = np.bincount(mat.row, minlength=self.userNum)
        order = np.argsort(-deg)
        self.templateUsers = order[: max(1, self.userNum // 4)]


        self.attackForm = "dataAttack"
        self.recommenderGradientRequired = False
        self.recommenderModelRequired = False


    def _pretrain_ae(self, epochs=10, batch=1024, lr=1e-3):
        opt = torch.optim.Adam(self.ae.parameters(), lr=lr)
        idx_all = np.arange(self.userNum)
        X = self.interact.tocsr()
        for ep in range(epochs):
            np.random.shuffle(idx_all)
            loss_sum, seen = 0.0, 0
            for i in range(0, len(idx_all), batch):
                ids = idx_all[i:i+batch]
                sub = X[ids].toarray().astype(np.float32)
                x = torch.from_numpy(sub).cuda()
                z = self.ae.encode(x)
                logits = self.ae.decode(z)
                loss = F.binary_cross_entropy_with_logits(logits, x)
                opt.zero_grad(); loss.backward(); opt.step()
                loss_sum += float(loss.item()) * len(ids); seen += len(ids)
            if (ep+1) % max(1, epochs//5) == 0:
                print(f"[AE] epoch {ep+1}/{epochs} loss={loss_sum/seen:.4f}")


    def _build_target_features(self, Eu):
        with torch.no_grad():
            _, Ei_prop = propagate_global(Eu, self.interact, K=self.K_prop)

        tgt = torch.tensor(self.targetItem, dtype=torch.long, device='cuda')
        e_t = Ei_prop.index_select(0, tgt)
        return e_t

    def _train_diffusion(self, epochs=300, batch=256, lr=1e-3):
        params = list(self.toa.parameters()) + list(self.diff.t_mlp.parameters())
        opt = torch.optim.Adam(params, lr=lr)


        self.ae.eval()


        with torch.no_grad():
            Eu = []
            step = 4096
            for i in range(0, self.userNum, step):
                x = torch.from_numpy(self.interact[i:i+step].toarray().astype(np.float32)).cuda()
                Eu.append(self.ae.encode(x))
            Eu = torch.cat(Eu, dim=0)


        e_t = self._build_target_features(Eu)


        user_pool = self.templateUsers

        for ep in range(epochs):
            np.random.shuffle(user_pool)
            loss_sum, seen = 0.0, 0

            for i in range(0, len(user_pool), batch):
                ids = user_pool[i:i+batch]
                eu0 = Eu[ids].detach()

                t = torch.randint(0, self.diff.T, (len(ids),), device='cuda').long()
                z_t, eps = self.diff.q_sample(eu0, t)


                t_emb = self.diff.t_embed(t)
                eps_hat = self.toa(z_t, t_emb, e_t)


                loss = F.mse_loss(eps_hat, eps)

                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()

                loss_sum += float(loss.item()) * len(ids); seen += len(ids)

            if (ep+1) % max(1, epochs//10) == 0:
                print(f"[ToDA] epoch {ep+1}/{epochs} Ld={loss_sum/seen:.4f}")


    def _sample_fake_batch(self, Eu, e_t, batch):
        B = batch
        z = torch.randn(B, self.zdim, device='cuda')
        for ts in reversed(range(self.diff.T)):
            t = torch.full((B,), ts, dtype=torch.long, device='cuda')
            t_emb = self.diff.t_embed(t)
            eps_hat = self.toa(z, t_emb, e_t)
            z = self.diff.p_step(z, eps_hat, t)
        logits = self.ae.decode(z)
        scores = torch.sigmoid(logits)
        return scores

    def _binarize(self, scores, k_each, targets):
        B, I = scores.shape
        non_target = np.array(sorted(set(range(I)) - set(targets)))
        out_rows = []
        sc = scores.detach().cpu().numpy()
        for r in range(B):
            topk = non_target[np.argsort(sc[r, non_target])[-k_each:]]
            chosen = set(topk.tolist()) | set(targets)
            rr = [0]*len(chosen); cc = list(chosen); vv = [1]*len(chosen)
            out_rows.append(csr_matrix((vv, (rr, cc)), shape=(1, I)))
        return out_rows


    def posionDataAttack(self,
                         epochs_ae=10, lr_ae=1e-3,
                         epochs_diff=300, lr_diff=1e-3,
                         batch=256):

        self._pretrain_ae(epochs=epochs_ae, lr=lr_ae, batch=batch)


        self._train_diffusion(epochs=epochs_diff, lr=lr_diff, batch=batch)


        self.ae.eval(); self.toa.eval()

        with torch.no_grad():
            Eu = []
            step = 4096
            for i in range(0, self.userNum, step):
                x = torch.from_numpy(self.interact[i:i+step].toarray().astype(np.float32)).cuda()
                Eu.append(self.ae.encode(x))
            Eu = torch.cat(Eu, dim=0)
            e_t = self._build_target_features(Eu)

        fake_rows = []
        remain = self.fakeUserNum
        while remain > 0:
            bs = min(batch, remain)
            scores = self._sample_fake_batch(Eu, e_t, bs)
            fake_rows.extend(self._binarize(scores, self.maliciousFeedbackNum, self.targetItem))
            remain -= bs

        poisoned = vstack([self.interact] + fake_rows)
        return poisoned
