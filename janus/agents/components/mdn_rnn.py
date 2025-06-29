import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


@dataclass

class MDNRNNConfig:
    latent_dim: int = 32
    action_dim: int = 4
    hidden_dim: int = 256
    num_mixtures: int = 5
    rnn_layers: int = 1
    temperature: float = 1.0
    use_layer_norm: bool = True
    rnn_type: str = "lstm"

class MDN_RNN(nn.Module):
    def __init__(self, config: MDNRNNConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.latent_dim + config.action_dim
        self.fc_input = nn.Linear(self.input_dim, config.hidden_dim)

        if config.rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(config.hidden_dim, config.hidden_dim, config.rnn_layers, batch_first=True)
        elif config.rnn_type.lower() == "gru":
            self.rnn = nn.GRU(config.hidden_dim, config.hidden_dim, config.rnn_layers, batch_first=True)
        else:
            raise ValueError(f"Unknown RNN type: {config.rnn_type}")

        self.layer_norm = nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity()

        self.fc_pi = nn.Linear(config.hidden_dim, config.num_mixtures)
        self.fc_mu = nn.Linear(config.hidden_dim, config.num_mixtures * config.latent_dim)
        self.fc_sigma = nn.Linear(config.hidden_dim, config.num_mixtures * config.latent_dim)

    def forward(self, z: torch.Tensor, a: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        batch_size, seq_len = z.shape[:2]
        x = torch.cat([z, a], dim=-1)
        x = self.fc_input(x)
        x = F.relu(x)
        rnn_out, hidden = self.rnn(x, hidden)
        rnn_out = self.layer_norm(rnn_out)

        pi_logits = self.fc_pi(rnn_out)
        pi = F.softmax(pi_logits / self.config.temperature, dim=-1)
        mu = self.fc_mu(rnn_out).view(batch_size, seq_len, self.config.num_mixtures, self.config.latent_dim)
        sigma = F.softplus(self.fc_sigma(rnn_out)).view(batch_size, seq_len, self.config.num_mixtures, self.config.latent_dim)
        return pi, mu, sigma, hidden

    def sample(self, pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        batch_size = pi.shape[0]
        categorical = Categorical(pi)
        mixture_idx = categorical.sample()
        mixture_idx_expanded = mixture_idx.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.config.latent_dim)
        mu_selected = torch.gather(mu, 1, mixture_idx_expanded).squeeze(1)
        sigma_selected = torch.gather(sigma, 1, mixture_idx_expanded).squeeze(1)
        normal = Normal(mu_selected, sigma_selected)
        return normal.sample()

    def loss(self, z_true: torch.Tensor, pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):
        batch_size = z_true.shape[0]
        z_true_expanded = z_true.unsqueeze(1).expand(-1, self.config.num_mixtures, -1)
        normal = Normal(mu, sigma)
        log_probs = normal.log_prob(z_true_expanded).sum(dim=-1)
        log_pi = torch.log(pi + 1e-8)
        log_probs = log_probs + log_pi
        max_log_probs = torch.max(log_probs, dim=1, keepdim=True)[0]
        log_sum_exp = max_log_probs + torch.log(torch.sum(torch.exp(log_probs - max_log_probs), dim=1, keepdim=True))
        nll = -log_sum_exp.squeeze()
        loss = nll.mean()
        loss_dict = {'mdn_loss': loss.item(), 'mean_nll': nll.mean().item(), 'std_nll': nll.std().item() if batch_size > 1 else 0.0}
        return loss, loss_dict

    def init_hidden(self, batch_size: int, device: torch.device):
        h = torch.zeros(self.config.rnn_layers, batch_size, self.config.hidden_dim, device=device)
        if self.config.rnn_type.lower() == "lstm":
            c = torch.zeros(self.config.rnn_layers, batch_size, self.config.hidden_dim, device=device)
            return (h, c)
        else:
            return h

class MDNRNNTrainer:
    def __init__(self, mdn_rnn: MDN_RNN, learning_rate: float = 1e-3):
        self.model = mdn_rnn
        self.optimizer = torch.optim.Adam(mdn_rnn.parameters(), lr=learning_rate)
        self.device = next(mdn_rnn.parameters()).device

    def train_step(self, z_seq: torch.Tensor, a_seq: torch.Tensor):
        batch_size, seq_len = z_seq.shape[:2]
        z_input = z_seq[:, :-1]
        a_input = a_seq[:, :-1]
        z_target = z_seq[:, 1:]
        hidden = self.model.init_hidden(batch_size, self.device)
        pi, mu, sigma, _ = self.model(z_input, a_input, hidden)
        total_loss = 0
        for t in range(seq_len - 1):
            loss, _ = self.model.loss(z_target[:, t].detach(), pi[:, t], mu[:, t], sigma[:, t])
            total_loss += loss
        total_loss = total_loss / (seq_len - 1)
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return {'loss': total_loss.item(), 'grad_norm': self._get_grad_norm()}

    def _get_grad_norm(self) -> float:
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
