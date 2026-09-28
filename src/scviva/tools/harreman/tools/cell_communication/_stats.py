"""Shared statistics helpers used by both cell-type-agnostic and cell-type-aware CCC scoring."""

from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


def z_to_pval_fdr(Z: np.ndarray, method: str = "fdr_bh") -> tuple[np.ndarray, np.ndarray]:
    """Convert an array of Z-scores to (p-values, FDR-corrected p-values).

    Z is expected to already be a numpy array (callers convert torch tensors
    to numpy via ``.detach().cpu().numpy()`` before calling this function,
    matching every existing call site's current behavior).
    """
    pvals = norm.sf(Z)
    flat_pvals = np.asarray(pvals).flatten()
    fdr_flat = multipletests(flat_pvals, method=method)[1]
    fdr = fdr_flat.reshape(np.asarray(pvals).shape)
    return pvals, fdr


PVAL_TIES = ("inclusive", "strict")


def check_pval_ties(pval_ties: str) -> str:
    """Validate the ``pval_ties`` option of the permutation tests."""
    if pval_ties not in PVAL_TIES:
        raise ValueError(f'"pval_ties" should be one of {list(PVAL_TIES)}, got {pval_ties!r}.')
    return pval_ties


def permutation_exceedances(perm: Any, observed: Any, pval_ties: str = "inclusive") -> Any:
    """Boolean mask of permuted scores that reach the observed score.

    The permutation p-value is ``(n_exceedances + 1) / (M + 1)``.

    - ``"inclusive"`` (default) counts ``perm >= observed``, the standard permutation p-value.
      A score that is matched by the null, e.g. a gene pair with no co-expression whose observed
      and permuted scores are all exactly 0, gets p = 1.
    - ``"strict"`` counts ``perm > observed``, the behavior of harreman / scviva-tools up to
      0.1.7. Ties are not counted, so an all-zero score and null gives the minimum p-value
      ``1 / (M + 1)``; kept only to reproduce earlier results.

    Works for numpy arrays and torch tensors (``observed`` must broadcast against ``perm``).
    """
    check_pval_ties(pval_ties)
    return perm >= observed if pval_ties == "inclusive" else perm > observed


def bh_over_tested(pvals: np.ndarray, tested: np.ndarray | None = None) -> np.ndarray:
    """Benjamini-Hochberg FDR over the entries marked as tested (all if ``tested`` is None).

    Untested entries get ``NaN``. Shapes are preserved.
    """
    pvals = np.asarray(pvals, dtype=float)
    if tested is None:
        return multipletests(pvals.flatten(), method="fdr_bh")[1].reshape(pvals.shape)
    tested = np.asarray(tested, dtype=bool)
    fdr = np.full(pvals.shape, np.nan)
    if tested.any():
        fdr[tested] = multipletests(pvals[tested], method="fdr_bh")[1]
    return fdr


def flatten(nested_list: list | tuple) -> Iterator[Any]:
    """Yield scalar values from a nested list or tuple structure."""
    for item in nested_list:
        if isinstance(item, list | tuple):
            yield from flatten(item)
        else:
            yield item


def compute_max_cs_gp(vals: torch.Tensor, node_degrees: torch.Tensor) -> torch.Tensor:
    """Compute max communication score for a single gene (vector)."""
    return 0.5 * torch.sum(node_degrees * vals**2)


def compute_max_cs(
    node_degrees: torch.Tensor, counts: torch.Tensor, gene_pairs_ind: list[tuple[Any, Any]]
) -> torch.Tensor:
    """Compute max communication scores per gene pair."""
    result = torch.empty(len(gene_pairs_ind), dtype=counts.dtype, device=counts.device)

    for i, (g1, _) in enumerate(gene_pairs_ind):
        if isinstance(g1, list):
            vals = counts[g1].mean(dim=0)
        else:
            vals = counts[g1]
        result[i] = compute_max_cs_gp(vals, node_degrees)

    return result
