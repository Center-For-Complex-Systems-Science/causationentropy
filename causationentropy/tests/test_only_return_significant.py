"""Regression tests for ``discover_network(only_return_significant=False)``."""

import pytest

from causationentropy.core.discovery import discover_network
from causationentropy.datasets.synthetic import linear_stochastic_gaussian_process


def _significant_edges(G):
    return sorted(
        (u, v, d["lag"]) for u, v, d in G.edges(data=True) if d.get("significant", True)
    )


@pytest.mark.parametrize("seed", [2, 6, 11])
def test_reporting_insignificant_links_keeps_significant_edges(seed):
    """Reporting insignificant links must not change which links are significant.

    The extra shuffle tests for the insignificant candidates draw from the same
    random generator as the selection itself. These seeds gave different
    significant edges when the reporting ran inside the per-target loop.
    """
    data, _ = linear_stochastic_gaussian_process(0.7, n=4, T=120, p=0.3, seed=seed)
    options = dict(
        method="standard",
        information="gaussian",
        max_lag=2,
        n_shuffles=50,
        random_state=seed,
    )

    G_default = discover_network(data, only_return_significant=True, **options)
    G_report = discover_network(data, only_return_significant=False, **options)

    assert _significant_edges(G_report) == _significant_edges(G_default)
    assert any(not d["significant"] for _, _, d in G_report.edges(data=True))
