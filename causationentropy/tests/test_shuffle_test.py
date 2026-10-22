from unittest.mock import patch

import numpy as np

from causationentropy.core.discovery import (
    _build_z_neighbors,
    _restricted_permutation,
    shuffle_test,
)
from causationentropy.core.information.conditional_mutual_information import (
    conditional_mutual_information,
)


def test_build_z_neighbors_stays_local_in_z_space():
    Z = np.array([[0.0], [0.1], [10.0], [10.1]])
    neighbors = _build_z_neighbors(Z, shuffle_neighbors=2)
    assert neighbors.shape == (4, 2)
    assert set(neighbors[0]).issubset({0, 1})
    assert set(neighbors[1]).issubset({0, 1})
    assert set(neighbors[2]).issubset({2, 3})
    assert set(neighbors[3]).issubset({2, 3})


def test_restricted_permutation_uses_only_allowed_donors():
    neighbors = np.array(
        [
            [0, 1],
            [1, 0],
            [2, 3],
            [3, 2],
        ],
        dtype=np.int64,
    )
    rng = np.random.default_rng(123)
    permutation = _restricted_permutation(neighbors, rng)
    assert permutation.shape == (4,)
    for sample_index, donor_index in enumerate(permutation):
        assert donor_index in neighbors[sample_index]


def test_shuffle_test_uses_local_permutation_when_z_is_present():
    X = np.arange(12, dtype=float).reshape(6, 2)
    Y = np.arange(6, dtype=float).reshape(-1, 1)
    Z = np.arange(6, dtype=float).reshape(-1, 1)
    local_permutation = np.array([1, 0, 3, 2, 5, 4])
    with patch(
        "causationentropy.core.discovery._restricted_permutation",
        return_value=local_permutation,
    ) as mock_restricted, patch(
        "causationentropy.core.discovery.conditional_mutual_information",
        return_value=0.0,
    ) as mock_cmi:
        shuffle_test(
            X,
            Y,
            Z,
            observed_cmi=1.0,
            n_shuffles=1,
            rng=42,
            shuffle_neighbors=2,
        )
    mock_restricted.assert_called_once()
    np.testing.assert_array_equal(mock_cmi.call_args.args[0], X[local_permutation, :])


def test_shuffle_test_uses_global_fallback_without_z():
    X = np.arange(12, dtype=float).reshape(6, 2)
    Y = np.arange(6, dtype=float).reshape(-1, 1)
    with patch(
        "causationentropy.core.discovery._restricted_permutation"
    ) as mock_restricted, patch(
        "causationentropy.core.discovery.conditional_mutual_information",
        return_value=0.0,
    ):
        shuffle_test(
            X,
            Y,
            None,
            observed_cmi=1.0,
            n_shuffles=3,
            rng=42,
        )
    mock_restricted.assert_not_called()


def test_shuffle_test_corrected_p_value_never_zero():
    X = np.arange(10, dtype=float).reshape(-1, 1)
    Y = np.arange(10, dtype=float).reshape(-1, 1)
    with patch(
        "causationentropy.core.discovery.conditional_mutual_information",
        return_value=0.0,
    ):
        result = shuffle_test(
            X,
            Y,
            None,
            observed_cmi=1.0,
            n_shuffles=9,
            rng=42,
        )
    assert result["P_value"] == 0.1
    assert result["P_value"] > 0.0


def test_shuffle_test_local_path_is_reproducible_with_same_seed():
    data_rng = np.random.default_rng(123)
    Z = data_rng.normal(size=(80, 2))
    X = 1.5 * Z[:, [0]] + data_rng.normal(size=(80, 1))
    Y = -0.8 * Z[:, [1]] + data_rng.normal(size=(80, 1))
    observed = conditional_mutual_information(X, Y, Z, method="gaussian")
    result_1 = shuffle_test(
        X,
        Y,
        Z,
        observed_cmi=observed,
        n_shuffles=20,
        rng=999,
        information="gaussian",
        shuffle_neighbors=5,
    )
    result_2 = shuffle_test(
        X,
        Y,
        Z,
        observed_cmi=observed,
        n_shuffles=20,
        rng=999,
        information="gaussian",
        shuffle_neighbors=5,
    )
    assert result_1 == result_2
