# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Vendored from lab_dev/rsl_rl@feature/amp (rsl_rl/storage/circular_buffer.py).
# Kept here because rsl-rl-lib 5.4.1's CircularBuffer lacks mini_batch_generator.

from __future__ import annotations

import torch
from collections.abc import Sequence


class CircularBuffer:
    """Circular buffer for storing a history of batched tensor data.

    The shape of the appended data is expected to be (batch_size, ...), where the first dimension is the
    batch dimension. Correspondingly, the shape of the ring buffer is (max_len, batch_size, ...).
    """

    def __init__(self, max_len: int, batch_size: int, device: str):
        if max_len < 1:
            raise ValueError(f"The buffer size should be greater than zero. However, it is set to {max_len}!")
        self._batch_size = batch_size
        self._device = device
        self._ALL_INDICES = torch.arange(batch_size, device=device)

        self._max_len = torch.full((batch_size,), max_len, dtype=torch.int, device=device)
        self._num_pushes = torch.zeros(batch_size, dtype=torch.long, device=device)
        # pointer to the current head of the circular buffer (-1 means not initialized)
        self._pointer: int = -1
        self._buffer: torch.Tensor = None  # type: ignore

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> str:
        return self._device

    @property
    def max_length(self) -> int:
        return int(self._max_len[0].item())

    @property
    def current_length(self) -> torch.Tensor:
        return torch.minimum(self._num_pushes, self._max_len)

    @property
    def buffer(self) -> torch.Tensor:
        """Complete circular buffer, shape [batch_size, max_length, data.shape[1:]]."""
        buf = self._buffer.clone()
        buf = torch.roll(buf, shifts=self.max_length - self._pointer - 1, dims=0)
        return torch.transpose(buf, dim0=0, dim1=1)

    def reset(self, batch_ids: Sequence[int] | None = None):
        if batch_ids is None:
            batch_ids = slice(None)
        self._num_pushes[batch_ids] = 0
        if self._buffer is not None:
            self._buffer[:, batch_ids, :] = 0.0

    def append(self, data: torch.Tensor):
        if data.shape[0] != self.batch_size:
            raise ValueError(f"Input data has '{data.shape[0]}' batch size, expecting '{self.batch_size}'")
        data = data.to(self._device)
        if self._buffer is None:
            self._pointer = -1
            self._buffer = torch.empty((self.max_length, *data.shape), dtype=data.dtype, device=self._device)
        self._pointer = (self._pointer + 1) % self.max_length
        self._buffer[self._pointer] = data
        is_first_push = self._num_pushes == 0
        if torch.any(is_first_push):
            self._buffer[:, is_first_push] = data[is_first_push]
        self._num_pushes += 1

    def __getitem__(self, key: torch.Tensor) -> torch.Tensor:
        if len(key) != self.batch_size:
            raise ValueError(f"'key' has length {len(key)}, expecting {self.batch_size}")
        if torch.any(self._num_pushes == 0) or self._buffer is None:
            raise RuntimeError("Attempting to retrieve from an empty circular buffer.")
        valid_keys = torch.minimum(key, self._num_pushes - 1)
        index_in_buffer = torch.remainder(self._pointer - valid_keys, self.max_length)
        return self._buffer[index_in_buffer, self._ALL_INDICES]

    def mini_batch_generator(self, fetch_length: int, num_mini_batches: int, num_epochs: int = 8):
        """Yield mini-batches sampled from the circular buffer.

        Each yielded batch has shape (mini_batch_size, ...) where data dims follow the appended tensor.
        """
        if torch.any(self._num_pushes == 0) or self._buffer is None:
            raise RuntimeError("Attempting to generate batches from an empty circular buffer.")
        min_current_length = torch.min(self.current_length).item()
        if fetch_length > min_current_length:
            raise ValueError(f"Fetch length {fetch_length} exceeds minimum current length {min_current_length}.")
        epoch_batch_size = self.batch_size * fetch_length
        mini_batch_size = epoch_batch_size // num_mini_batches
        if epoch_batch_size % num_mini_batches != 0:
            raise ValueError(f"Epoch batch size {epoch_batch_size} is not divisible by {num_mini_batches} mini-batches.")

        total_combinations = self.current_length[0] * self.batch_size
        linear_indices = torch.randperm(total_combinations, device=self.device)[:epoch_batch_size]
        indices_0 = linear_indices // self.batch_size
        indices_1 = linear_indices % self.batch_size

        for _ in range(num_epochs):
            indices = torch.randperm(epoch_batch_size, requires_grad=False, device=self.device)
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                mini_indices_0 = indices_0[indices[start:end]]
                mini_indices_1 = indices_1[indices[start:end]]
                yield self._buffer[mini_indices_0, mini_indices_1]
