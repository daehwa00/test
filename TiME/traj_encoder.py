import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from TiME.backbones import build_backbone

class _MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        norm: str,
        ef_enabled: bool,
        mr_enabled: bool,
        reset_dt: float,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ef_enabled = ef_enabled
        self.mamba = build_backbone(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            ef_enabled=ef_enabled,
            mr_enabled=mr_enabled,
            reset_dt=reset_dt,
        )

    def forward(self, seq, resets=None):
        normalized = self.norm(seq)
        if self.ef_enabled:
            return seq + self.mamba(normalized, resets=resets)
        return seq + self.mamba(normalized)

    def step(self, seq, conv_state, ssm_state, resets=None):
        # Inference
        res, new_conv_state, new_ssm_state = self.mamba.step(
            self.norm(seq), conv_state, ssm_state
        )
        return seq + res, new_conv_state, new_ssm_state


class _MambaHiddenState:
    def __init__(self, conv_states: list[torch.Tensor], ssm_states: list[torch.Tensor]):
        assert len(conv_states) == len(ssm_states)
        self.n_layers = len(conv_states)
        self.conv_states = conv_states
        self.ssm_states = ssm_states

    def reset(self, idxs):
        for i in range(self.n_layers):
            # hidden states are initialized to zero
            self.conv_states[i][idxs] = 0.0
            self.ssm_states[i][idxs] = 0.0

    def __getitem__(self, layer_idx: int):
        assert layer_idx < self.n_layers
        return self.conv_states[layer_idx], self.ssm_states[layer_idx]

    def __setitem__(self, layer_idx: int, conv_ssm: tuple[torch.Tensor]):
        conv, ssm = conv_ssm
        self.conv_states[layer_idx] = conv
        self.ssm_states[layer_idx] = ssm

    def reset_state(self, done: torch.Tensor):
        """
        Reset the states for environments where `done` is True.
        Args:
            done (torch.Tensor): Boolean tensor indicating environments to reset.
        """
        # Ensure `done` is a Boolean tensor
        done = done.bool()
        for layer_idx in range(len(self.conv_states)):
            # Reset convolutional state
            conv_state = self.conv_states[layer_idx]
            if isinstance(conv_state, torch.Tensor):
                conv_state[done] = 0  # Reset only for `done` environments
            # Reset SSM state
            ssm_state = self.ssm_states[layer_idx]
            if isinstance(ssm_state, torch.Tensor):
                ssm_state[done] = 0  # Reset only for `done` environments


class MambaTrajEncoder(nn.Module):
    def __init__(
        self,
        tstep_dim: int,
        max_seq_len: int,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 2,
        norm: str = "layer",
        ef_enabled: bool = True,
        mr_enabled: bool = True,
        reset_dt: float = 5.0,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.inp = nn.Linear(tstep_dim, d_model)

        self.mambas = nn.ModuleList(
            [
                _MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    norm=norm,
                    ef_enabled=ef_enabled,
                    mr_enabled=mr_enabled,
                    reset_dt=reset_dt,
                )
                for _ in range(n_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self._emb_dim = d_model

    def init_hidden_state(self, batch_size: int, device: torch.device):
        conv_states, ssm_states = [], []
        for mamba_block in self.mambas:
            conv_state, ssm_state = mamba_block.mamba.allocate_inference_cache(
                batch_size, max_seqlen=self.max_seq_len
            )
            conv_states.append(conv_state.to(device))
            ssm_states.append(ssm_state.to(device))
        return _MambaHiddenState(conv_states, ssm_states)

    def reset_state(self, hidden_state, dones):
        if hidden_state is None:
            return None
        assert isinstance(hidden_state, _MambaHiddenState)
        hidden_state.reset(idxs=dones)
        return hidden_state

    def forward(self, seq, hidden_state=None, resets=None):
        seq = self.inp(seq)
        if hidden_state is None:
            # Training
            for mamba in self.mambas:
                seq = mamba(seq, resets=resets)
        else:
            # Inference
            assert not self.training
            assert isinstance(hidden_state, _MambaHiddenState)
            for i, mamba in enumerate(self.mambas):
                conv_state_i, ssm_state_i = hidden_state[i]
                seq, new_conv_state_i, new_ssm_state_i = mamba.step(
                    seq, conv_state=conv_state_i, ssm_state=ssm_state_i
                )
                hidden_state[i] = new_conv_state_i, new_ssm_state_i
        return self.out_norm(seq), hidden_state

    @property
    def emb_dim(self):
        return self._emb_dim
