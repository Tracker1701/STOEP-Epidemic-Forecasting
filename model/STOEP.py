import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttentionEnhanceBlock(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        channels: int,
        heads: int = 4,
        tau: float = 1.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        fusion_mode: str = 'conv',
        gate_type: str = 'scalar',
        symmetrize: bool = True,
        normalize: str = 'softmax',
        eps: float = 1e-6
    ):
        super().__init__()
        assert channels % heads == 0
        assert fusion_mode in ('conv', 'gate')
        assert normalize in ('softmax', 'row')
        self.num_nodes = num_nodes
        self.channels = channels
        self.heads = heads
        self.dk = channels // heads
        self.scale = (self.dk ** 0.5) * max(tau, 1e-6)
        self.symmetrize = symmetrize
        self.normalize = normalize
        self.eps = eps
        self.q_linear = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_linear = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)
        self.structural_attn = nn.Parameter(torch.eye(num_nodes), requires_grad=True)
        self.fusion_mode = fusion_mode
        if fusion_mode == 'conv':
            self.fusion = nn.Conv2d(2, 1, kernel_size=1, bias=True)
        else:
            if gate_type == 'scalar':
                self.gate = nn.Parameter(torch.tensor(0.0))
            elif gate_type == 'vector':
                self.gate = nn.Parameter(torch.zeros(num_nodes))
            else:
                raise ValueError("gate_type must be 'scalar' or 'vector'")
            self.gate_type = gate_type
        self.softmax_last = nn.Softmax(dim=-1)

    def _row_normalize(self, A: torch.Tensor) -> torch.Tensor:
        denom = A.sum(-1, keepdim=True).clamp_min(self.eps)
        return A / denom

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N, T = x.size()
        assert N == self.num_nodes and C == self.channels
        Q = self.q_linear(x)
        K = self.k_linear(x)
        Q_ = Q.permute(0, 3, 2, 1).contiguous().view(B * T, N, self.heads, self.dk).permute(0, 2, 1, 3)
        K_ = K.permute(0, 3, 2, 1).contiguous().view(B * T, N, self.heads, self.dk).permute(0, 2, 1, 3)
        att_scores = torch.matmul(Q_, K_.transpose(-1, -2)) / self.scale
        att_node = F.softmax(att_scores, dim=-1)
        att_node = self.attn_drop(att_node)
        att_node = att_node.view(B, T, self.heads, N, N).mean(dim=(1, 2))
        att_struct = F.softmax(self.structural_attn, dim=-1).unsqueeze(0).expand(B, -1, -1)
        if self.fusion_mode == 'conv':
            fusion_input = torch.stack([att_node, att_struct], dim=1)
            fused = self.fusion(fusion_input).squeeze(1)
        else:
            if self.gate_type == 'scalar':
                g = torch.sigmoid(self.gate)
                fused = g * att_node + (1.0 - g) * att_struct
            else:
                g_vec = torch.sigmoid(self.gate)
                g = g_vec.view(1, N, 1)
                fused = g * att_node + (1.0 - g) * att_struct
        if self.symmetrize:
            fused = 0.5 * (fused + fused.transpose(1, 2))
        if self.normalize == 'softmax':
            fused = F.softmax(fused, dim=-1)
        else:
            fused = self._row_normalize(fused)
        fused = self.attn_drop(fused)
        x_bt = x.permute(0, 3, 2, 1)
        x_att = torch.einsum('bij,btjc->btic', fused, x_bt)
        x_att = self.proj_drop(x_att)
        x_out = x_att.permute(0, 3, 2, 1).contiguous()
        return x_out

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()
    def forward(self, x, A):
        if len(A.shape) == 2:
            x = torch.einsum('vw, ncwl->ncvl', A, x)
        else:
            x = torch.einsum('nvw, ncwl->ncvl', A, x)
        return x.contiguous()

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=1)
    def forward(self, x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=2, order=1):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = order * support_len * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order
    def forward(self, x, support):
        out = []
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return F.dropout(h, self.dropout, training=self.training)

class SPE(nn.Module):
    def __init__(
        self, num_nodes, dropout, in_dim, out_len,
        residual_channels, dilation_channels, skip_channels,
        end_channels, kernel_size, blocks, layers
    ):
        super(SPE, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.residual_channels = residual_channels
        self.out_len = out_len
        self.start_conv = nn.Conv2d(in_dim, residual_channels, kernel_size=1)
        receptive_field = 1
        self.supports_len = 2
        self.saeb_attn = SpatialAttentionEnhanceBlock(
            num_nodes, residual_channels,
            heads=4, tau=1.0,
            attn_dropout=0.1, proj_dropout=0.1,
            fusion_mode='gate', gate_type='scalar',
            symmetrize=True, normalize='softmax'
        )
        self.attn_weight = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.ln = nn.ModuleList()
        self.gconv = nn.ModuleList()
        for b in range(blocks):
            additional_scope = 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(residual_channels, dilation_channels,
                              kernel_size=(1, kernel_size), dilation=(1, new_dilation))
                )
                self.gate_convs.append(
                    nn.Conv2d(residual_channels, dilation_channels,
                              kernel_size=(1, kernel_size), dilation=(1, new_dilation))
                )
                self.residual_convs.append(nn.Conv2d(dilation_channels, residual_channels, kernel_size=1))
                self.skip_convs.append(nn.Conv2d(dilation_channels, skip_channels, kernel_size=1))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                T_len = (2 ** layers - 1) * blocks + 2 - receptive_field
                self.ln.append(nn.LayerNorm([residual_channels, num_nodes, T_len]))
                self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))
        self.end_conv_b1 = nn.Conv2d(skip_channels * blocks * layers, end_channels, kernel_size=1)
        self.end_conv_b2 = nn.Conv2d(end_channels, out_len, kernel_size=1)
        self.end_conv_g1 = nn.Conv2d(skip_channels * blocks * layers, end_channels, kernel_size=1)
        self.end_conv_g2 = nn.Conv2d(end_channels, out_len, kernel_size=1)
        self.receptive_field = receptive_field

    def forward(self, input, adp_g):
        b, _, _, in_len = input.size()
        if in_len < self.receptive_field:
            x = F.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        x_att = self.saeb_attn(x)
        x = x + self.attn_weight * (x_att - x)
        skip = None
        dense = None
        for i in range(self.blocks * self.layers):
            res = x
            f = torch.tanh(self.filter_convs[i](x))
            g = torch.sigmoid(self.gate_convs[i](x))
            x = f * g
            s = self.skip_convs[i](x)
            skip = s if skip is None else torch.cat((s, skip[:, :, :, -s.size(3):]), dim=1)
            x = self.gconv[i](x, adp_g)
            if dense is None:
                dense = res[:, :, :, -x.size(3):]
            else:
                dense = dense[:, :, :, -x.size(3):]
            gate2 = torch.sigmoid(x)
            x = x * gate2 + dense * (1 - gate2)
            x = self.ln[i](x)
        param_b = torch.sigmoid(self.end_conv_b2(F.relu(self.end_conv_b1(F.relu(skip)))))
        param_g = torch.sigmoid(self.end_conv_g2(F.relu(self.end_conv_g1(F.relu(skip)))))
        return param_b, param_g

class CAA(nn.Module):
    def __init__(self, num_nodes: int, in_dim: int, win_len: int,
                 num_patterns: int = 8, d_model: int = 64, d_affine: int = 32):
        super().__init__()
        self.N = num_nodes
        self.C = in_dim
        self.S = win_len
        self.P = num_patterns
        self.d = d_model
        self.dA = d_affine
        self.pattern_keys = nn.Parameter(torch.randn(self.P, self.S) * 0.01)
        self.seq_embed   = nn.Linear(self.S, self.d, bias=True)
        self.key_embed   = nn.Linear(self.S, self.d, bias=True)
        self.value_embed = nn.Linear(self.S, self.d, bias=True)
        self.row_proj = nn.Linear(self.d, self.dA, bias=True)
        self.col_emb  = nn.Parameter(torch.randn(self.N, self.dA) * 0.01)
        self.scale = nn.Parameter(torch.zeros(1))
        self.attn_drop = nn.Dropout(0.0)
        self.proj_drop = nn.Dropout(0.0)
        self.eps = 1e-6

    def _zscore_last(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std  = x.std(dim=-1, keepdim=True).clamp_min(self.eps)
        return (x - mean) / std

    def forward(self, x_node: torch.Tensor) -> torch.Tensor:
        B, C, N, T = x_node.shape
        assert C == self.C and N == self.N and T >= self.S
        if self.scale.item() == 0.0:
            return x_node.new_zeros((B, N, N))
        x_tail = x_node[..., -self.S:]
        x_win  = x_tail[:, 1, :, :]
        x_win  = self._zscore_last(x_win)
        q = self.seq_embed(x_win)
        k = self.key_embed(self.pattern_keys)
        v = self.value_embed(self.pattern_keys)
        k = k.unsqueeze(0).expand(B, -1, -1)
        v = v.unsqueeze(0).expand(B, -1, -1)
        attn_scores = torch.matmul(q, k.transpose(1, 2)) / (self.d ** 0.5)
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.attn_drop(attn)
        h = torch.matmul(attn, v)
        h = self.proj_drop(h)
        row = self.row_proj(h)
        delta = torch.einsum('bid,jd->bij', row, self.col_emb)
        return self.scale * delta

class FMF(nn.Module):
    def __init__(self,
                 adaptive_thresholds: bool = True,
                 qI: float = 0.10,
                 qB: float = 0.20,
                 qG: float = 0.20,
                 qZR: float = 0.90,
                 ema_beta: float = 0.90,
                 i_eps_min: float = 1e-8,
                 b_thr_min: float = 2e-3,
                 g_thr_min: float = 2e-3,
                 qzr_min: float = 0.70,
                 qzr_max: float = 0.98,
                 zi_scale: float = 0.5):
        super().__init__()
        self.adaptive_thresholds = adaptive_thresholds
        self.qI, self.qB, self.qG, self.qZR = qI, qB, qG, qZR
        self.ema_beta = ema_beta
        self.i_eps_min, self.b_thr_min, self.g_thr_min = i_eps_min, b_thr_min, g_thr_min
        self.qzr_min, self.qzr_max = qzr_min, qzr_max
        _INIT_QI = 1e-8
        _INIT_QB = 2e-2
        _INIT_QG = 2e-2
        _INIT_QZR = 0.90
        self.register_buffer("running_qI", torch.tensor(_INIT_QI))
        self.register_buffer("running_qB", torch.tensor(_INIT_QB))
        self.register_buffer("running_qG", torch.tensor(_INIT_QG))
        self.register_buffer("running_qZR", torch.tensor(_INIT_QZR))
        self.zi_scale = zi_scale

    @staticmethod
    def _quantile_safe(x: torch.Tensor, q: float) -> torch.Tensor:
        x_flat = x.reshape(-1).float()
        if x_flat.numel() == 0:
            return torch.tensor(0.0, device=x.device, dtype=torch.float32)
        q = min(max(q, 0.0), 1.0)
        return torch.quantile(x_flat, q)

    def _update_ema(self, name: str, value: torch.Tensor):
        old = getattr(self, name)
        new = self.ema_beta * old + (1 - self.ema_beta) * value
        setattr(self, name, new.detach())

    @staticmethod
    def _small_param_mask(param_b_t: torch.Tensor, param_g_t: torch.Tensor,
                          b_thr: torch.Tensor, g_thr: torch.Tensor):
        if param_b_t.dim() == 3:
            b_max = param_b_t.max(dim=-1).values
        else:
            b_max = param_b_t
        if param_g_t.dim() == 3:
            g_max = param_g_t.max(dim=-1).values
        else:
            g_max = param_g_t
        small_mask = (b_max <= b_thr) & (g_max <= g_thr)
        return small_mask

    def _compute_quiet_mask_adaptive(self, SIR_hist: torch.Tensor, device):
        I_hist = SIR_hist[..., 1]
        curr_qI = self._quantile_safe(I_hist, self.qI)
        if self.training:
            self._update_ema("running_qI", curr_qI)
        i_abs_eps = torch.clamp(self.running_qI, min=self.i_eps_min)
        zero_ratio = (I_hist <= i_abs_eps).float().mean(dim=1)
        curr_qZR = self._quantile_safe(zero_ratio, self.qZR)
        curr_qZR = torch.clamp(curr_qZR, min=self.qzr_min, max=self.qzr_max)
        if self.training:
            self._update_ema("running_qZR", curr_qZR)
        quiet_zero_ratio = self.running_qZR
        quiet_mask = (zero_ratio >= quiet_zero_ratio).to(device=device)
        return quiet_mask, i_abs_eps, quiet_zero_ratio

    def _compute_param_thresholds_adaptive(self, param_b_t: torch.Tensor, param_g_t: torch.Tensor):
        b_base = param_b_t if param_b_t.dim() == 2 else param_b_t.max(dim=-1).values
        g_base = param_g_t if param_g_t.dim() == 2 else param_g_t.max(dim=-1).values
        curr_qB = self._quantile_safe(b_base, self.qB)
        curr_qG = self._quantile_safe(g_base, self.qG)
        if self.training:
            self._update_ema("running_qB", curr_qB)
            self._update_ema("running_qG", curr_qG)
        b_thr = torch.clamp(self.running_qB, min=self.b_thr_min)
        g_thr = torch.clamp(self.running_qG, min=self.g_thr_min)
        return b_thr, g_thr

    def forward(
        self,
        param_b_t: torch.Tensor,
        param_g_t: torch.Tensor,
        mob_t: torch.Tensor,
        SIR_curr: torch.Tensor,
        SIR_hist: torch.Tensor = None
    ):
        device = SIR_curr.device
        if self.adaptive_thresholds and (SIR_hist is not None):
            quiet_mask, i_abs_eps, quiet_zero_ratio = self._compute_quiet_mask_adaptive(SIR_hist, device)
            b_thr, g_thr = self._compute_param_thresholds_adaptive(param_b_t, param_g_t)
        else:
            _FALLBACK_QZR = torch.tensor(0.90, device=device)
            _FALLBACK_BTHR = torch.tensor(2e-2, device=device)
            _FALLBACK_GTHR = torch.tensor(2e-2, device=device)
            _FALLBACK_IEPS = torch.tensor(1e-8, device=device)
            b_, T_hist, N_, three_ = (SIR_hist.shape if SIR_hist is not None else (SIR_curr.shape[0], 1, SIR_curr.shape[1], 3))
            quiet_mask = torch.zeros((b_, N_), dtype=torch.bool, device=device)
            i_abs_eps = _FALLBACK_IEPS
            quiet_zero_ratio = _FALLBACK_QZR
            b_thr = _FALLBACK_BTHR
            g_thr = _FALLBACK_GTHR
        if mob_t.dim() == 2:
            mob_t = mob_t.unsqueeze(0).expand(SIR_curr.size(0), -1, -1)
        num_node = SIR_curr.size(-2)
        S = SIR_curr[..., [0]]
        I = SIR_curr[..., [1]]
        R = SIR_curr[..., [2]]
        pop = (S + I + R).expand(-1, num_node, num_node)
        propagation = (mob_t / pop * I.expand(-1, num_node, num_node)).sum(1) + \
                      (mob_t / pop * I.expand(-1, num_node, num_node).transpose(1, 2)).sum(2)
        propagation = propagation.unsqueeze(2)
        small_mask = self._small_param_mask(param_b_t, param_g_t, b_thr, g_thr)
        force_zero_mask = (quiet_mask | small_mask).unsqueeze(-1)
        if param_b_t.dim() == 3:
            param_b_t = param_b_t * (~force_zero_mask)
        else:
            param_b_t = param_b_t * (~force_zero_mask)
        if param_g_t.dim() == 3:
            param_g_t = param_g_t * (~force_zero_mask)
        else:
            param_g_t = param_g_t * (~force_zero_mask)
        I_new = param_b_t * propagation
        if I_new.dim() == 3:
            I_new = I_new * (~force_zero_mask)
        else:
            I_new = I_new * (~force_zero_mask)
        R_t = I * param_g_t + R
        I_t = I + I_new - I * param_g_t
        S_t = S - I_new
        I_t = torch.where(force_zero_mask, I + self.zi_scale * I * param_b_t - I * param_g_t, I_t)
        I_new_floor = (self.zi_scale * I * param_b_t).detach()
        I_new = torch.where(force_zero_mask, I_new_floor, I_new)
        if I_new.dim() == 3 and I_new.size(-1) != 1:
            I_new_col = I_new[..., -1:]
        else:
            I_new_col = I_new if I_new.dim() == 3 else I_new.unsqueeze(-1)
        return torch.cat((I_new_col, S_t, I_t, R_t), dim=-1)

class stoep(nn.Module):
    def __init__(
        self, num_nodes,
        dropout=0.5, in_dim=4, in_len=14, out_len=14,
        residual_channels=32, dilation_channels=32,
        skip_channels=256, end_channels=512, kernel_size=2,
        blocks=2, layers=3,
        caa_num_patterns: int = 9,
        caa_d_model: int = 64,
        caa_d_affine: int = 32,
        adaptive_thresholds: bool = True,
        qI: float = 0.10, qB: float = 0.20, qG: float = 0.20, qZR: float = 0.90,
        ema_beta: float = 0.90,
        i_eps_min: float = 0.5, b_thr_min: float = 2e-3, g_thr_min: float = 2e-3,
        qzr_min: float = 0.70, qzr_max: float = 0.98,
        od_scale_factor: float = 3.0
    ):
        super().__init__()
        self.SPE = SPE(
            num_nodes, dropout, in_dim, out_len,
            residual_channels, dilation_channels,
            skip_channels, end_channels, kernel_size,
            blocks, layers
        )
        self.FMF  = FMF(
            adaptive_thresholds=adaptive_thresholds,
            qI=qI, qB=qB, qG=qG, qZR=qZR,
            ema_beta=ema_beta,
            i_eps_min=i_eps_min, b_thr_min=b_thr_min, g_thr_min=g_thr_min,
            qzr_min=qzr_min, qzr_max=qzr_max
        )
        self.out_dim  = out_len
        self.in_len   = in_len
        self.num_nodes = num_nodes
        self.in_dim   = in_dim
        self.od_scale_factor = od_scale_factor
        self.inc_init = nn.Parameter(torch.empty(out_len, in_len))
        nn.init.normal_(self.inc_init, 1, 0.01)
        self.caa_dyn = CAA(
            num_nodes=num_nodes,
            in_dim=in_dim,
            win_len=in_len,
            num_patterns=caa_num_patterns,
            d_model=caa_d_model,
            d_affine=caa_d_affine
        )

    def forward(self, x_node, SIR, od, max_od):
        SIR_hist = SIR
        SIR_curr0 = SIR[:, -1, ...]
        incidence = torch.softmax(self.inc_init, dim=1)
        mob = torch.einsum('kl,blnmc->bknmc', incidence, od).squeeze(-1)
        g   = mob.mean(1)
        delta_g = self.caa_dyn(x_node)
        g = g + delta_g
        g_t  = g.permute(0, 2, 1)
        g_dyn = [g / g.sum(2, True).clamp_min(1e-6), g_t / g_t.sum(2, True).clamp_min(1e-6)]
        param_b, param_g = self.SPE(x_node, g_dyn)
        outputs = []
        SIR_t = SIR_curr0
        for i in range(self.out_dim):
            NSIR = self.FMF(
                param_b[:, i], param_g[:, i],
                mob[:, i] * max_od * self.od_scale_factor,
                SIR_t, SIR_hist=SIR_hist
            )
            SIR_t = NSIR[..., 1:]
            outputs.append(NSIR[..., [0]])
        return torch.stack(outputs, dim=1)
