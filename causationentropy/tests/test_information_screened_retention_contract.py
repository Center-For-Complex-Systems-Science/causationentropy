import numpy as np

import causationentropy.core.discovery as discovery


def test_information_screened_retention_is_soft_and_preserves_proposal(monkeypatch):
    X = np.column_stack(
        [
            np.linspace(0.0, 1.0, 24),
            np.linspace(1.0, 2.0, 24),
            np.linspace(2.0, 3.0, 24),
            np.linspace(3.0, 4.0, 24),
            np.linspace(4.0, 5.0, 24),
        ]
    )
    Y = np.linspace(-1.0, 1.0, 24).reshape(-1, 1)
    Z_init = np.linspace(10.0, 11.0, 24).reshape(-1, 1)

    # The proposal already exceeds the nominal 40% target (3 > ceil(0.4 * 5)=2).
    # The screened path must preserve it rather than truncating it.
    monkeypatch.setattr(
        discovery,
        "information_lasso_optimal_causation_entropy",
        lambda *args, **kwargs: [0, 1, 2],
    )

    rescue_conditioning = []

    def fake_cmi(x, _y, z, **kwargs):
        rescue_conditioning.append(z.copy())
        # Candidate 4 wins the single rescue slot over candidate 3.
        return 2.0 if np.isclose(x[0, 0], 4.0) else 1.0

    monkeypatch.setattr(discovery, "conditional_mutual_information", fake_cmi)

    captured = {}

    def fake_standard(x, _y, z_init, _rng, **kwargs):
        captured["screened"] = x.copy()
        captured["z_init"] = z_init.copy()
        return []

    monkeypatch.setattr(
        discovery,
        "standard_optimal_causation_entropy",
        fake_standard,
    )

    selected = discovery.information_screened_optimal_causation_entropy(
        X,
        Y,
        Z_init,
        np.random.default_rng(0),
        retention=0.40,
    )

    assert selected == []
    np.testing.assert_allclose(captured["screened"], X[:, [0, 1, 2, 4]])
    np.testing.assert_allclose(captured["z_init"], Z_init)

    expected_conditioning = np.hstack((Z_init, X[:, [0, 1, 2]]))
    assert len(rescue_conditioning) == 2
    for z in rescue_conditioning:
        np.testing.assert_allclose(z, expected_conditioning)
