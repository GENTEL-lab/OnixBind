# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Utils for geometry library."""

from collections.abc import Iterable
import numbers

import numpy as np

def unstack(value: np.ndarray, axis: int = -1) -> list[np.ndarray]:
  return [
      np.squeeze(v, axis=axis)
      for v in np.split(value, value.shape[axis], axis=axis)
  ]


def angdiff(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
  """Compute absolute difference between two angles."""
  d = alpha - beta
  d = (d + np.pi) % (2 * np.pi) - np.pi
  return d


def safe_arctan2(
    x1: np.ndarray, x2: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
  """Safe version of arctan2 that avoids NaN gradients when x1=x2=0."""

  return safe_select(
      np.abs(x1) + np.abs(x2) < eps,
      lambda: np.zeros_like(np.arctan2(x1, x2)),
      lambda: np.arctan2(x1, x2),
  )


def weighted_mean(
    *,
    weights: np.ndarray,
    value: np.ndarray,
    axis: int | Iterable[int] | None = None,
    eps: float = 1e-10,
) -> np.ndarray:
  """Computes weighted mean in a safe way that avoids NaNs.

  This is equivalent to np.average for the case eps=0.0, but adds a small
  constant to the denominator of the weighted average to avoid NaNs.
  'weights' should be broadcastable to the shape of value.

  Args:
    weights: Weights to weight value by.
    value: Values to average
    axis: Axes to average over.
    eps: Epsilon to add to the denominator.

  Returns:
    Weighted average.
  """

  weights = np.asarray(weights, dtype=value.dtype)
  weights = np.broadcast_to(weights, value.shape)

  weights_shape = weights.shape

  if isinstance(axis, numbers.Integral):
    axis = [axis]
  elif axis is None:
    axis = list(range(len(weights_shape)))

  return np.sum(weights * value, axis=axis) / (
      np.sum(weights, axis=axis) + eps
  )
