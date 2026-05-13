from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

import PIL
import torchvision.transforms as transforms
import my_transforms as my_transforms
from time import time
import logging

import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.distributions import Normal, Independent

def reverse_kl(student_logits, teacher_logits, T: float = 1.0) -> torch.Tensor:
    # KL(p_s || p_t)，对学生参数可导；教师 logits 不需要梯度
    if T is None or T <= 0:
        T = 1.0
    s = student_logits / T
    t = teacher_logits.detach() / T            # 教师不回传梯度
    log_p_s = F.log_softmax(s, dim=1)
    log_p_t = F.log_softmax(t, dim=1)          # 数值稳定
    p_s = log_p_s.exp()
    kl = (p_s * (log_p_s - log_p_t)).sum(dim=1).mean()
    return kl * (T ** 2)

class AugPolicy(nn.Module):
    def __init__(self, in_dim=2, hidden=32, init_std=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, 3)       # 三个强度：geom、color、noise
        self.logstd = nn.Parameter(torch.full((3,), float(torch.log(torch.tensor(init_std)))))
    def forward(self, state):  # state: [B, in_dim] or [in_dim] (用 batch 均值即可)
        if state.dim() == 1: state = state.unsqueeze(0)
        h = self.net(state)
        mu = self.mu_head(h)                      # 形状 [B,3]
        std = self.logstd.exp().expand_as(mu)     # 固定对角高斯
        dist = Independent(Normal(mu, std), 1)
        z = dist.rsample()                        # reparameterized sample
        a = torch.tanh(z) * 0.5 + 0.5            # 映射到 (0,1)
        logp = dist.log_prob(z) - torch.sum(torch.log(1 - torch.tanh(z)**2 + 1e-6), dim=-1)  # tanh 修正
        ent = dist.base_dist.entropy().sum(-1)   # 近似熵
        return a, logp, ent

def param_augment(x, a, img_size=384, device=None):
    # a: [B,3] in (0,1) -> 解析成强度
    if device is None:
        device = x.device
    B, C, H, W = x.shape
    a = a.to(x.dtype).to(device)

    # 几何强度（小角度、小平移、小尺度）
    geom = a[:, 0]   # 0~1
    max_deg = 15.0; min_deg = 0.0
    degrees = (geom * (max_deg - min_deg) + min_deg)
    max_trans = 1/16
    trans = geom * max_trans
    max_scale_jit = 0.1
    scale = 1.0 + (geom - 0.5) * 2.0 * max_scale_jit

    # 颜色强度
    color = a[:, 1]
    def lerp_color(strength, lo, hi):  # 把 0~1 映到对称抖动区间
        span = (hi - lo) * strength
        return 1.0 - span/2.0, 1.0 + span/2.0
    b_lo, b_hi = lerp_color(color, 0.0, 0.4)     # brightness
    c_lo, c_hi = lerp_color(color, 0.0, 0.3)     # contrast
    s_lo, s_hi = lerp_color(color, 0.0, 0.5)     # saturation
    hue_max = 0.06 * color

    # 模糊强度
    sigma = 0.001 + color * (0.5 - 0.001)
    # 噪声强度
    noise = a[:, 2]
    noise_std = 0.0 + noise * 0.02

    outputs = []
    for i in range(B):
        xi = x[i]

        # 几何：先 pad 再 affine 再 center crop
        pad = int(H/2)
        xi = F.pad(xi.unsqueeze(0), (pad, pad, pad, pad), mode='replicate').squeeze(0)
        angle = float(degrees[i].item())
        translate = [
            float((W + 2*pad) * trans[i].item()),
            float((H + 2*pad) * trans[i].item())
        ]
        xi = TF.affine(xi, angle=angle, translate=translate, scale=float(scale[i].item()), shear=[0.0, 0.0])
        xi = TF.center_crop(xi, [H, W])

        # 水平翻转：随机数与 xi 同 device，转成 float 比较
        if float(torch.rand((), device=xi.device).item()) < float(geom[i].item()):
            xi = torch.flip(xi, dims=[2])  # 宽度维度

        # 颜色抖动：端点先转 float，再用 [0,1) 采样线性插值，避免 uniform_ 的张量端点
        blo = float(b_lo[i].item());  bhi = float(b_hi[i].item())
        clo = float(c_lo[i].item());  chi = float(c_hi[i].item())
        slo = float(s_lo[i].item());  shi = float(s_hi[i].item())
        hm  = float(hue_max[i].item())

        u = torch.rand((), device=xi.device).item()
        br = blo + u * (bhi - blo)
        u = torch.rand((), device=xi.device).item()
        ct = clo + u * (chi - clo)
        u = torch.rand((), device=xi.device).item()
        st = slo + u * (shi - slo)

        xi = TF.adjust_brightness(xi, br)
        xi = TF.adjust_contrast (xi, ct)
        xi = TF.adjust_saturation(xi, st)

        if hm > 1e-6:
            u = torch.rand((), device=xi.device).item()
            hue = (u * 2.0 - 1.0) * hm   # 在 [-hm, hm] 采样
            xi = TF.adjust_hue(xi, float(hue))

        # 模糊
        xi = TF.gaussian_blur(xi, kernel_size=5, sigma=float(sigma[i].item()))

        # 噪声
        ns = float(noise_std[i].item())
        if ns > 0:
            xi = torch.clamp(xi + torch.randn_like(xi) * ns, 0.0, 1.0)

        outputs.append(xi)

    return torch.stack(outputs, dim=0).to(device)

def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False):
    img_shape = (384, 384, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        my_transforms.Clip(0.0, 1.0), 
        my_transforms.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),  
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            resample=PIL.Image.BILINEAR,
            fillcolor=None
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        my_transforms.GaussianNoise(0, gaussian_std),
        my_transforms.Clip(clip_min, clip_max)
    ])
    return tta_transforms


def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


class CoTTA(nn.Module):
    """CoTTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(
        self,
        model,
        optimizer,
        steps=1,
        episodic=False,
        mt_alpha=0.995,
        rst_m=0.1,
        ap=0.9,
        opd_weight=1,
        opd_temp=2,
        opd_aug_views=2,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.model_ema.eval()  # 教师稳定性

        # RL-OPD 相关
        self.rl_opd = True
        self.policy = AugPolicy(in_dim=2, hidden=32, init_std=0.5).to(next(model.parameters()).device)
        self.policy_baseline = torch.zeros((), device=next(model.parameters()).device)  # 标量 EMA 基线
        self.policy_beta = 1e-3   # 策略熵系数
        self.policy_lambda = 0.1  # 强度惩罚系数 λ ||a||^2
        self.policy_m = 0.9       # 奖励基线动量
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=1e-4)

        self.transform = get_tta_transforms()    
        self.mt = mt_alpha
        self.rst = rst_m
        self.ap = ap
        self.opd_weight = float(opd_weight)
        self.opd_temp = float(opd_temp)
        self.opd_aug_views = int(opd_aug_views)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # Use this line to also restore the teacher model                         
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.model_ema.eval()


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        # outputs, stu_loss_high, stu_loss_low = self.model(x,counter,corruption_type)
        outputs = self.model(x)
        # Teacher Prediction
        # anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max(1)[0]
        # standard_ema, _,_ = self.model_ema(x)
        with torch.no_grad():
            standard_ema = self.model_ema(x)
        outputs_ema = standard_ema
        # Student update
        proximal_term = 0.0      

        # for (name1,param1),(name2,param2) in zip(self.model.named_parameters(),self.model_ema.named_parameters()):
        #     if "adaptmlp" in name1:
        #         proximal_term += (param1 - param2).norm(2)
        # # Base CoTTA consistency loss (student matches EMA teacher on current batch).
        # loss = (softmax_entropy(outputs, outputs_ema.detach())).mean(0) + (1/2)*proximal_term
        loss = (softmax_entropy(outputs, outputs_ema.detach())).mean(0)

        optimizer.zero_grad()
        loss.backward()

        # 严格 OPD + RL：策略决定增广强度，师生在同一 x_aug 上对齐
        rl_logs = []  # 存每个视图的 (logp, entropy, reward)
        self.opd_weight = 1.0
        self.opd_aug_views = 1
        if self.opd_weight > 0.0 and self.opd_aug_views > 0:
            scale = self.opd_weight / float(self.opd_aug_views)
            T = self.opd_temp
            # state: 用干净样本的统计（不建图）
            with torch.no_grad():
                prob_clean = torch.softmax(outputs.detach(), dim=1)
                ent = -(prob_clean * torch.log(prob_clean + 1e-8)).sum(dim=1).mean()
                conf = prob_clean.max(dim=1).values.mean()
                state = torch.stack([ent, conf], dim=0)

            for _ in range(self.opd_aug_views):
                if self.rl_opd:
                    a, logp, ent_pi = self.policy(state)           # 需要图用于 REINFORCE
                    a_b = a.detach().expand(x.shape[0], -1)        # 关键：断开到学生损失的路径梯度
                    with torch.no_grad():                           # 关键：增广不建图，纯 on-policy
                        x_aug = param_augment(x, a_b, img_size=x.shape[-1], device=x.device)
                else:
                    x_aug = self.transform(x)
                    logp = ent_pi = None

                # 学生前向（带梯度）
                student_aug = self.model(x_aug)
                # 教师前向（无梯度）
                with torch.no_grad():
                    teacher_aug = self.model_ema(x_aug)

                # 反向 KL（学生 || 教师）
                kl_rev = reverse_kl(student_aug, teacher_aug, T=T)
                (scale * kl_rev).backward()

                if self.rl_opd:
                    strength = (a.pow(2).mean())
                    reward = (-kl_rev.detach()) - self.policy_lambda * strength
                    rl_logs.append((logp.squeeze(0), ent_pi.squeeze(0), reward))

                    print(reward)

        # 学生参数更新
        optimizer.step()
        optimizer.zero_grad()

        # 策略更新（REINFORCE 带 EMA 基线 + 熵正则），不回传到学生/教师
        if self.rl_opd and len(rl_logs) > 0:
            logps = torch.stack([lp for lp, _, _ in rl_logs])
            ents  = torch.stack([en for _, en, _ in rl_logs])
            rewards = torch.stack([rw for _, _, rw in rl_logs])
            with torch.no_grad():
                b = self.policy_baseline
                adv = rewards - b
                # 更新 EMA 基线
                self.policy_baseline = self.policy_m * b + (1 - self.policy_m) * rewards.mean()
            policy_loss = -(adv.detach() * logps).mean() - self.policy_beta * ents.mean()
            self.policy_opt.zero_grad(set_to_none=True)
            policy_loss.backward()
            self.policy_opt.step()
        # Teacher update
        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=self.mt)
        # Stochastic restore
        if True:
            for nm, m  in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape, device=p.device) < self.rst).float()
                        with torch.no_grad():
                            p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1.-mask)
        return outputs_ema


@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if True:#isinstance(m, nn.BatchNorm2d): collect all 
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    # print(nm, np)
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what we update
    # model.requires_grad_(False)
    # enable all trainable
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        else:
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
