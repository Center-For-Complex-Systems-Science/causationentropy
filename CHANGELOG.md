## [0.1.0] - 2025-09-15
### Added
- All math code parsed to new API

## [1.0.0] - 2025-10-01
### Added
- Full test coverage.
- Full API design

## [1.1.0] - 2025-11-12
### Added
- Added support for converting causal networks to pandas DataFrames.
- Added utility for importing and processing Tigramite network structures.
- Added companion matrix calculation for better FPR/TPR calculations.
- Added plot_causal_network function that implements some automated graph layout algorithms to plot causal networks.
- Enforce non-negative conditional mututal information and non-negative mutual information.

## [Unreleased]
### Added
- `random_state` parameter for `discover_network` to seed the permutation tests (#26).
- `only_return_significant` parameter for `discover_network`; with `False`, tested but non-significant links are returned as well and `network_to_dataframe` adds a `Significant` column (#36).
- `plot_delay_analysis` for plotting CMI against lag for each directed pair (#36).
- `linear_gaussian_from_graph` for generating synthetic data from a chosen `networkx` graph (#39).
- Block-bootstrap confidence intervals: `moving_block_bootstrap_indices`, `stationary_bootstrap_indices`, `bootstrap_confidence_interval` and `bootstrap_cmi_confidence_interval` (#40).
- Optional futility early stopping in `shuffle_test` (`early_stop=True`). It is only available when calling `shuffle_test` directly (#41).
- Multiple-testing corrections `bonferroni_correction`, `benjamini_hochberg_correction`, `benjamini_yekutieli_correction`, `adaptive_bh_correction` and `estimate_null_proportion`, plus `apply_test_correction` to correct the p-values of a discovered network after discovery (adds a `P_Adjusted` column to `network_to_dataframe`). Discovery itself does not apply a correction (#37).
- Tutorials and expanded theory documentation (#30, #49, #50).

### Changed
- NumPy 2 is now required (#34), so Python 3.8 is no longer supported.
- Tigramite is now a development-only dependency, installed with `pip install causationentropy[dev]` (#24, #35).
- The k-NN conditional mutual information estimator uses Chebyshev distance by default; `metric` now defaults to `None` in `discover_network` and `conditional_mutual_information`, which picks each estimator's own default. This changes results for `information="knn"` (#48).
- `causationentropy` no longer imports its `tests` package (#34).

### Fixed
- `Compute_TPR_FPR` no longer counts diagonal entries as false positives, which could push the FPR above 1. This changes reported FPR values (#38).
- `kde_entropy` now stays in log space, so low-density samples no longer turn the entropy into `-inf` (#42).
- The conditional Poisson estimator now uses full covariance submatrices (`np.ix_`) instead of only their diagonal entries, which changes its values (#42). The sign change made in the same PR broke Poisson discovery and is reverted in #51.
- Compatibility with current NumPy and Matplotlib releases (#25).
