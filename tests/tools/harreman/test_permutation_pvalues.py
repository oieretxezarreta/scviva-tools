"""Tests for the permutation p-value tie handling, the optional expression filter of the cell
type-aware test, per-sample standardization on a cell subset, and device resolution."""

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData
from statsmodels.stats.multitest import multipletests

from scviva.tools import HarremanAnalysis
from scviva.tools.harreman.tools.cell_communication import (
    compute_cell_communication,
    compute_ct_cell_communication,
    compute_ct_interacting_cell_scores,
    compute_gene_pairs,
    compute_interacting_cell_scores,
    select_significant_interactions,
)
from scviva.tools.harreman.tools.cell_communication._stats import (
    bh_over_tested,
    check_pval_ties,
    permutation_exceedances,
)

M = 50


def _adata(cell_types=("A", "B"), n_obs=60, zero_gene0_in=None, sample_ids=None, seed=0):
    """8 genes (2 LR pairs), a deterministic spatial layout and cell types assigned in turn."""
    gene_names = [f"gene{i}" for i in range(8)]
    rng = np.random.default_rng(seed)
    X = rng.poisson(2.0, size=(n_obs, len(gene_names))).astype(float)
    labels = [cell_types[i % len(cell_types)] for i in range(n_obs)]
    if zero_gene0_in is not None:
        X[np.array(labels) == zero_gene0_in, 0] = 0.0  # gene0 never expressed in this cell type
    obs = pd.DataFrame(
        {"cell_type": pd.Categorical(labels)}, index=[f"cell{i}" for i in range(n_obs)]
    )
    if sample_ids is not None:
        obs["sample_id"] = sample_ids
    adata = AnnData(
        X=X,
        obs=obs,
        var=pd.DataFrame(index=gene_names),
        obsm={"spatial": rng.uniform(0, 100, size=(n_obs, 2))},
    )
    database = pd.DataFrame(0.0, index=gene_names, columns=["pair0", "pair1"])
    database.loc["gene0", "pair0"] = 1.0
    database.loc["gene1", "pair0"] = -1.0
    database.loc["gene2", "pair1"] = 1.0
    database.loc["gene3", "pair1"] = -1.0
    adata.varm["database"] = database
    adata.uns["database"] = "LR"
    adata.uns["ligand"] = pd.DataFrame({0: ["gene0", "gene2"]}, index=["pair0", "pair1"])
    adata.uns["receptor"] = pd.DataFrame({0: ["gene1", "gene3"]}, index=["pair0", "pair1"])
    return adata


def _prepare_ct(adata, cell_type_pairs=None, sample_key=None):
    HarremanAnalysis(adata).tl.compute_knn_graph(
        compute_neighbors_on_key="spatial", n_neighbors=5, sample_key=sample_key
    )
    compute_gene_pairs(
        adata,
        cell_type_key="cell_type",
        cell_type_pairs=cell_type_pairs,
        ct_specific=True,
        verbose=False,
    )
    return adata


def _run_ct(adata, **kw):
    kw = {
        "model": "none",
        "cell_type_key": "cell_type",
        "M": M,
        "seed": 0,
        "test": "non-parametric",
        "device": "cpu",
        "verbose": False,
        **kw,
    }
    compute_ct_cell_communication(adata, **kw)
    return adata.uns["ct_ccc_results"]["np"]


# ---------------------------------------------------------------- helpers


def test_permutation_exceedances_counts_ties_only_when_inclusive():
    perm = np.array([[0.0, 0.0, 1.0], [2.0, 3.0, 3.0]])
    obs = np.array([[0.0], [3.0]])
    np.testing.assert_array_equal(permutation_exceedances(perm, obs).sum(1), [3, 2])
    np.testing.assert_array_equal(permutation_exceedances(perm, obs, "strict").sum(1), [1, 0])
    t = permutation_exceedances(torch.tensor(perm), torch.tensor(obs)).sum(dim=1)
    np.testing.assert_array_equal(t.numpy(), [3, 2])


def test_check_pval_ties_rejects_unknown_values():
    assert check_pval_ties("strict") == "strict"
    with pytest.raises(ValueError):
        check_pval_ties("ties")


def test_bh_over_tested_ignores_untested_entries():
    p = np.array([[0.01, 0.5, 0.02], [0.03, 0.9, 0.04]])
    tested = np.array([[True, False, True], [True, False, False]])
    fdr = bh_over_tested(p, tested)
    assert np.isnan(fdr[~tested]).all()
    np.testing.assert_allclose(fdr[tested], multipletests(p[tested], method="fdr_bh")[1])
    np.testing.assert_allclose(
        bh_over_tested(p), multipletests(p.ravel(), method="fdr_bh")[1].reshape(p.shape)
    )


# ---------------------------------------------------------------- cell type-aware test


def test_zero_score_with_zero_null_gets_p_one_by_default():
    res = _run_ct(_prepare_ct(_adata(zero_gene0_in="A")))
    cs, perm = res["gp"]["cs"], res["gp"]["perm_cs"]
    degenerate = (cs == 0) & (perm == 0).all(axis=2)
    assert degenerate.any()
    np.testing.assert_array_equal(res["gp"]["pval"][degenerate], 1.0)


def test_strict_reproduces_the_previous_p_values_and_only_p_changes():
    strict = _run_ct(_prepare_ct(_adata(zero_gene0_in="A")), pval_ties="strict")
    inclusive = _run_ct(_prepare_ct(_adata(zero_gene0_in="A")), pval_ties="inclusive")
    for level in ("gp", "m"):
        np.testing.assert_array_equal(strict[level]["cs"], inclusive[level]["cs"])
        np.testing.assert_array_equal(strict[level]["perm_cs"], inclusive[level]["perm_cs"])
        cs, perm = strict[level]["cs"], strict[level]["perm_cs"]
        # the formula used up to scviva-tools 0.1.7
        old = (np.sum(perm > cs[:, :, None], axis=2) + 1) / (M + 1)
        np.testing.assert_array_equal(strict[level]["pval"], old)
        new = (np.sum(perm >= cs[:, :, None], axis=2) + 1) / (M + 1)
        np.testing.assert_array_equal(inclusive[level]["pval"], new)
    degenerate = (strict["gp"]["cs"] == 0) & (strict["gp"]["perm_cs"] == 0).all(axis=2)
    np.testing.assert_allclose(strict["gp"]["pval"][degenerate], 1 / (M + 1))


def test_expression_filter_marks_untested_pairs_and_restricts_bh():
    adata = _prepare_ct(_adata(zero_gene0_in="A"))
    res = _run_ct(adata, min_cells_per_cell_type=1)
    tested = res["gp"]["tested"]
    genes = adata.uns["genes"]
    cell_types = adata.obs["cell_type"].astype(str).values
    counts = adata[:, genes].X
    for k, (ct1, ct2) in enumerate(adata.uns["cell_type_pairs"]):
        for j, (i1, i2) in enumerate(adata.uns["gene_pairs_ind"]):
            ok1 = (counts[cell_types == ct1][:, i1] > 0).sum() >= 1
            ok2 = (counts[cell_types == ct2][:, i2] > 0).sum() >= 1
            assert tested[k, j] == (ok1 and ok2)
    assert (~tested).any()
    assert tested.any()
    assert np.isnan(res["gp"]["pval"][~tested]).all()
    assert np.isnan(res["gp"]["FDR"][~tested]).all()
    np.testing.assert_allclose(
        res["gp"]["FDR"][tested], multipletests(res["gp"]["pval"][tested], method="fdr_bh")[1]
    )
    df = adata.uns["ct_ccc_results"]["cell_com_df_gp"]
    np.testing.assert_array_equal(df["tested_np"].values, tested.ravel())


def test_expression_filter_can_exclude_everything():
    res = _run_ct(_prepare_ct(_adata()), min_cells_per_cell_type=10_000)
    for level in ("gp", "m"):
        assert not res[level]["tested"].any()
        assert np.isnan(res[level]["FDR"]).all()


def test_no_filter_by_default():
    res = _run_ct(_prepare_ct(_adata()))
    assert "tested" not in res["gp"]
    assert not np.isnan(res["gp"]["FDR"]).any()


# ---------------------------------------------------------------- per-sample standardization


def test_sample_key_with_a_cell_type_subset_runs():
    # a third cell type is not part of the analysed pair, so only a subset of cells is used;
    # this raised IndexError in standardize_ct_counts before
    samples = ["s1"] * 45 + ["s2"] * 45
    adata = _prepare_ct(
        _adata(cell_types=("A", "B", "C"), n_obs=90, sample_ids=samples),
        cell_type_pairs=[("A", "B")],
        sample_key="sample_id",
    )
    _run_ct(adata, model="danb", test="both")
    assert len(adata.uns["cells"]) < adata.n_obs


def test_single_constant_sample_matches_no_sample_key():
    kw = {"cell_types": ("A", "B", "C"), "n_obs": 90}
    a = _prepare_ct(_adata(**kw), cell_type_pairs=[("A", "B")])
    b = _prepare_ct(
        _adata(**kw, sample_ids=["s1"] * 90), cell_type_pairs=[("A", "B")], sample_key="sample_id"
    )
    ra = _run_ct(a, model="danb", test="both")
    rb = _run_ct(b, model="danb", test="both")
    np.testing.assert_allclose(
        a.uns["ct_ccc_results"]["p"]["gp"]["cs"], b.uns["ct_ccc_results"]["p"]["gp"]["cs"]
    )
    np.testing.assert_allclose(ra["gp"]["pval"], rb["gp"]["pval"])


# ---------------------------------------------------------------- default device


def test_ct_interacting_cell_scores_default_device():
    adata = _prepare_ct(_adata())
    _run_ct(adata, test="both")
    select_significant_interactions(
        adata, ct_aware=True, test="non-parametric", use_FDR=False, threshold=1.01
    )
    compute_ct_interacting_cell_scores(
        adata, test="both"
    )  # device="auto" used to reach torch unresolved
    assert "ct_interacting_cell_results" in adata.uns


def test_interacting_cell_scores_default_device_and_pval_ties():
    adata = _adata()
    HarremanAnalysis(adata).tl.compute_knn_graph(compute_neighbors_on_key="spatial", n_neighbors=5)
    compute_gene_pairs(adata, ct_specific=False, verbose=False)
    compute_cell_communication(
        adata, model="none", M=M, seed=0, test="both", device="cpu", pval_ties="strict"
    )
    assert adata.uns["pval_ties"] == "strict"
    select_significant_interactions(adata, test="non-parametric", use_FDR=False, threshold=1.01)
    compute_interacting_cell_scores(adata, test="both", M=5)  # default device
    assert "interacting_cell_results" in adata.uns
    with pytest.raises(ValueError):
        compute_interacting_cell_scores(adata, test="both", M=5, pval_ties="nope")


# ---------------------------------------------------------------- HarremanAnalysis wrapper


def test_harreman_analysis_passes_options_through():
    adata = _adata(zero_gene0_in="A")
    ha = HarremanAnalysis(adata)
    ha.tl.compute_knn_graph(compute_neighbors_on_key="spatial", n_neighbors=5)
    ha._completed_steps.add("setup")
    adata.uns["harreman"]["cell_type_key"] = "cell_type"
    ha.compute_gene_pairs()
    ha.compute_cell_communication(
        mode="cell_type",
        model="none",
        n_permutations=M,
        seed=0,
        test="non-parametric",
        device="cpu",
        pval_ties="strict",
        min_cells_per_cell_type=1,
    )
    assert adata.uns["pval_ties"] == "strict"
    assert adata.uns["min_cells_per_cell_type"] == 1
    assert "tested" in adata.uns["ct_ccc_results"]["np"]["gp"]
    with pytest.raises(ValueError):
        ha.compute_cell_communication(
            mode="standard",
            model="none",
            n_permutations=M,
            device="cpu",
            min_cells_per_cell_type=1,
        )
