import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix, vstack

from util.tool import targetItemSelect


class _WMFSurrogate(nn.Module):
    def __init__(self, n_users, n_items, dim):
        super().__init__()
        self.user_emb = nn.Parameter(torch.randn(n_users, dim) * 0.1)
        self.item_emb = nn.Parameter(torch.randn(n_items, dim) * 0.1)

    def forward(self):
        return self.user_emb @ self.item_emb.t()


class _ItemAESurrogate(nn.Module):
    def __init__(self, n_users, hidden_dims):
        super().__init__()
        q_dims = [n_users] + list(hidden_dims)
        p_dims = q_dims[::-1]
        self.q_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(q_dims[:-1], q_dims[1:])]
        )
        self.p_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(p_dims[:-1], p_dims[1:])]
        )

    def forward(self, inputs):
        h = inputs
        for layer in self.q_layers:
            h = torch.tanh(layer(h))
        for idx, layer in enumerate(self.p_layers):
            h = layer(h)
            if idx != len(self.p_layers) - 1:
                h = torch.tanh(h)
        return h


def _weighted_mf_loss(data_tensor, user_emb, item_emb, weight_alpha, l2_reg):
    scores = user_emb @ item_emb.t()
    weights = torch.ones_like(data_tensor) + weight_alpha * data_tensor
    recon_loss = (weights * (data_tensor - scores) ** 2).mean()
    reg_loss = l2_reg * (user_emb.pow(2).mean() + item_emb.pow(2).mean())
    return recon_loss + reg_loss


def _itemae_loss(data_tensor, logits, weight_alpha):
    weights = torch.ones_like(data_tensor)
    weights[data_tensor > 0] = weight_alpha
    return (weights * (data_tensor - logits) ** 2).sum(dim=1).mean()


def _target_ce_loss(logits, target_items):
    if len(target_items) == 0:
        return logits.new_tensor(0.0)
    log_probs = torch.log_softmax(logits, dim=1)
    target_mask = torch.zeros_like(logits)
    target_mask[:, target_items] = 1.0
    per_user_targets = target_mask.sum(dim=1).clamp_min(1.0)
    return (-(log_probs * target_mask).sum(dim=1) / per_user_targets).mean()


class AIA:
    """
    Lightweight port of revisit_adv_rec's black-box adversarial fake-data attack.
    Uses a WMF surrogate and differentiable inner updates to optimize fake users.
    """
    def __init__(self, arg, data):
        self.data = data
        self.interact = data.matrix().tocsr()
        self.userNum = self.interact.shape[0]
        self.itemNum = self.interact.shape[1]

        self.targetItem = targetItemSelect(data, arg)
        self.targetItem = [data.item[i.strip()] for i in self.targetItem]

        self.attackForm = "dataAttack"
        self.recommenderGradientRequired = False
        self.recommenderModelRequired = False

        self.maliciousUserSize = arg.maliciousUserSize
        self.maliciousFeedbackSize = arg.maliciousFeedbackSize
        if self.maliciousFeedbackSize == 0:
            self.maliciousFeedbackNum = int(self.interact.sum() / data.user_num)
        elif self.maliciousFeedbackSize >= 1:
            self.maliciousFeedbackNum = int(self.maliciousFeedbackSize)
        else:
            self.maliciousFeedbackNum = int(self.maliciousFeedbackSize * self.itemNum)

        if self.maliciousUserSize < 1:
            self.fakeUserNum = int(data.user_num * self.maliciousUserSize)
        else:
            self.fakeUserNum = int(self.maliciousUserSize)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and getattr(arg, "use_cuda", True) else "cpu"
        )

        self.adv_epochs = getattr(arg, "advAttackEpochs", 30)
        self.surrogate_epochs = getattr(arg, "advAttackSurrogateEpochs", 20)
        self.unroll_steps = getattr(arg, "advAttackUnrollSteps", 1)
        self.adv_lr = getattr(arg, "advAttackLr", 1.0)
        self.adv_momentum = getattr(arg, "advAttackMomentum", 0.95)
        self.proj_threshold = getattr(arg, "advAttackProjThreshold", 0.1)
        self.surrogate_dim = getattr(arg, "advAttackDim", 64)
        self.surrogate_lr = getattr(arg, "advAttackSurrogateLr", 1e-2)
        self.surrogate_l2 = getattr(arg, "advAttackSurrogateL2", 1e-5)
        self.weight_alpha = getattr(arg, "advAttackWeightAlpha", 20.0)
        self.click_targets = getattr(arg, "advAttackClickTargets", False)
        self.max_init_click = getattr(arg, "advAttackMaxInitClick", 100)
        self.attack_type = getattr(arg, "advAttackType", "adversarial")
        self.surrogate_model = getattr(arg, "advAttackSurrogateModel", "wmf_sgd")
        self.surrogate_batch_size = getattr(arg, "advAttackBatchSize", 2048)
        self.itemae_hidden_dims = getattr(arg, "advAttackItemAEHiddenDims", "256,128")
        self.itemae_hidden_dims = [int(x) for x in str(self.itemae_hidden_dims).split(",") if str(x).strip()]

    def _init_fake_data(self):
        train_dense = self.interact.toarray().astype(np.float32)
        user_clicks = np.asarray(train_dense.sum(1)).reshape(-1)
        qual_users = np.where(user_clicks <= self.max_init_click)[0]
        if len(qual_users) < self.fakeUserNum:
            qual_users = np.arange(self.userNum)
        sampled_users = np.random.choice(qual_users, size=self.fakeUserNum, replace=len(qual_users) < self.fakeUserNum)
        return train_dense[sampled_users].copy()

    def _project_fake_tensor(self, fake_tensor):
        projected = (fake_tensor > self.proj_threshold).float()
        if self.click_targets and len(self.targetItem) > 0:
            projected[:, self.targetItem] = 1.0
        if self.maliciousFeedbackNum > 0:
            keep_n = min(self.maliciousFeedbackNum, self.itemNum)
            topk_idx = torch.topk(fake_tensor, k=keep_n, dim=1).indices
            projected.zero_()
            projected.scatter_(1, topk_idx, 1.0)
            if self.click_targets and len(self.targetItem) > 0:
                projected[:, self.targetItem] = 1.0
        return projected

    def _fit_wmf_adv(self, data_tensor):
        try:
            import higher
        except ImportError as exc:
            raise ImportError(
                "AIA now expects the `higher` package for differentiable inner-loop training."
            ) from exc

        train_tensor = data_tensor
        model = _WMFSurrogate(train_tensor.shape[0], self.itemNum, self.surrogate_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.surrogate_lr)

        warmup_epochs = max(0, self.surrogate_epochs - self.unroll_steps)
        for _ in range(warmup_epochs):
            loss = _weighted_mf_loss(
                train_tensor,
                model.user_emb,
                model.item_emb,
                self.weight_alpha,
                self.surrogate_l2,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with higher.innerloop_ctx(model, optimizer, copy_initial_weights=False) as (fmodel, diffopt):
            for _ in range(max(1, self.unroll_steps)):
                loss = _weighted_mf_loss(
                    train_tensor,
                    fmodel.user_emb,
                    fmodel.item_emb,
                    self.weight_alpha,
                    self.surrogate_l2,
                )
                diffopt.step(loss)

            predictions = fmodel()
            adv_loss = _target_ce_loss(predictions[:self.userNum], self.targetItem)
            adv_grads = torch.autograd.grad(adv_loss, data_tensor)[0]
            return adv_loss, adv_grads[-self.fakeUserNum:, :]

    def _fit_itemae_adv(self, data_tensor):
        try:
            import higher
        except ImportError as exc:
            raise ImportError(
                "AIA now expects the `higher` package for differentiable inner-loop training."
            ) from exc

        data_transposed = data_tensor.t()
        model = _ItemAESurrogate(data_tensor.shape[0], self.itemae_hidden_dims).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.surrogate_lr, weight_decay=self.surrogate_l2)

        n_rows = data_transposed.shape[0]
        batch_size = self.surrogate_batch_size if self.surrogate_batch_size > 0 else n_rows
        row_indices = np.arange(n_rows)

        warmup_epochs = max(0, self.surrogate_epochs - self.unroll_steps)
        for _ in range(warmup_epochs):
            np.random.shuffle(row_indices)
            for start in range(0, n_rows, batch_size):
                batch_idx = row_indices[start:start + batch_size]
                batch_tensor = data_transposed[batch_idx]
                outputs = model(batch_tensor)
                loss = _itemae_loss(batch_tensor, outputs, self.weight_alpha)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with higher.innerloop_ctx(model, optimizer, copy_initial_weights=False) as (fmodel, diffopt):
            for _ in range(max(1, self.unroll_steps)):
                np.random.shuffle(row_indices)
                for start in range(0, n_rows, batch_size):
                    batch_idx = row_indices[start:start + batch_size]
                    batch_tensor = data_transposed[batch_idx]
                    outputs = fmodel(batch_tensor)
                    loss = _itemae_loss(batch_tensor, outputs, self.weight_alpha)
                    diffopt.step(loss)

            preds = []
            for start in range(0, n_rows, batch_size):
                batch_idx = row_indices[start:start + batch_size]
                preds.append(fmodel(data_transposed[batch_idx]))
            predictions = torch.cat(preds, dim=0).t()
            adv_loss = _target_ce_loss(predictions[:self.userNum], self.targetItem)
            adv_grads = torch.autograd.grad(adv_loss, data_tensor)[0]
            return adv_loss, adv_grads[-self.fakeUserNum:, :]

    def _compute_adv_loss_and_grads(self, data_tensor):
        if self.surrogate_model == "wmf_sgd":
            return self._fit_wmf_adv(data_tensor)
        if self.surrogate_model == "item_ae":
            return self._fit_itemae_adv(data_tensor)
        raise ValueError("Unknown surrogate model {} for AIA.".format(self.surrogate_model))

    def posionDataAttack(self):
        real_tensor = torch.tensor(self.interact.toarray(), dtype=torch.float32, device=self.device)
        fake_init = self._init_fake_data()
        fake_tensor = torch.tensor(fake_init, dtype=torch.float32)
        fake_tensor.requires_grad_()
        optimizer = torch.optim.SGD([fake_tensor], lr=self.adv_lr, momentum=self.adv_momentum)

        if self.attack_type == "random":
            random_fake = torch.randn_like(fake_tensor)
            projected = self._project_fake_tensor(random_fake)
            fake_np = projected.detach().cpu().numpy().astype(np.float32)
            fake_rat = csr_matrix(fake_np, shape=(self.fakeUserNum, self.itemNum), dtype=np.float32)
            return vstack([self.interact, fake_rat])

        for epoch in range(self.adv_epochs):
            optimizer.zero_grad()
            fake_current = fake_tensor.detach().clone().to(self.device).requires_grad_(True)
            data_tensor = torch.cat([real_tensor, fake_current], dim=0)
            adv_loss, adv_grads = self._compute_adv_loss_and_grads(data_tensor)
            row_norm = adv_grads.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
            normalized_grads = adv_grads / row_norm
            fake_tensor.grad = normalized_grads.detach().cpu()
            optimizer.step()
            projected = self._project_fake_tensor(fake_tensor.data.to(self.device)).detach().cpu()
            fake_tensor.data = projected
            print(
                "AIA epoch {}: adv_loss={:.6f}, active_entries={:.0f}".format(
                    epoch + 1, adv_loss.item(), fake_tensor.sum().item()
                )
            )

        fake_np = fake_tensor.detach().cpu().numpy().astype(np.float32)
        fake_rat = csr_matrix(fake_np, shape=(self.fakeUserNum, self.itemNum), dtype=np.float32)
        return vstack([self.interact, fake_rat])
