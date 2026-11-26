"""Reusable correlation blocks for Gaussian conditional MI surrogates."""

import numpy as np

from causationentropy.core.information.conditional_mutual_information import (
    gaussian_conditional_mutual_information,
)


class GaussianShuffleCache:
    """Cache the fixed Y/Z correlation block during a Gaussian shuffle test."""

    def __init__(self, X, Y, Z, x_scaled, yz_scaled, corr_x, corr_yz):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.x_scaled = x_scaled
        self.yz_scaled = yz_scaled
        self.corr_x = corr_x
        self.corr_yz = corr_yz
        self.n, self.kx = X.shape
        self.ky = Y.shape[1]
        self.kz = Z.shape[1]
        self._zidx = np.arange(self.ky, self.ky + self.kz)
        self._xzidx = np.r_[
            np.arange(self.kx),
            np.arange(self.kx + self.ky, self.kx + self.ky + self.kz),
        ]
        self._s_yz = np.linalg.slogdet(corr_yz)[1]
        self._s_z = np.linalg.slogdet(corr_yz[np.ix_(self._zidx, self._zidx)])[1]

    @classmethod
    def build(cls, X, Y, Z):
        """Build the cache when the fixed covariance blocks are well conditioned."""
        if Z is None:
            return None
        X, Y, Z = (np.asarray(v) for v in (X, Y, Z))
        if any(v.ndim != 2 for v in (X, Y, Z)):
            return None
        n = len(X)
        if n < 3 or any(len(v) != n or v.shape[1] == 0 for v in (X, Y, Z)):
            return None
        if any(not np.all(np.isfinite(v)) for v in (X, Y, Z)):
            return None
        yz = np.column_stack((Y, Z))
        std_x = np.std(X, axis=0, ddof=1)
        std_yz = np.std(yz, axis=0, ddof=1)
        if np.any(std_x < 1e-12) or np.any(std_yz < 1e-12):
            return None
        x_scaled = (X - np.mean(X, axis=0)) / std_x
        yz_scaled = (yz - np.mean(yz, axis=0)) / std_yz
        corr_x = (x_scaled.T @ x_scaled) / (n - 1)
        corr_yz = (yz_scaled.T @ yz_scaled) / (n - 1)
        # Preserve the original calculation for ill-conditioned matrices.
        if (
            np.min(np.linalg.eigvalsh(corr_x)) <= 1e-9
            or np.min(np.linalg.eigvalsh(corr_yz)) <= 1e-9
        ):
            return None
        return cls(X, Y, Z, x_scaled, yz_scaled, corr_x, corr_yz)

    def _reference(self, indices):
        return gaussian_conditional_mutual_information(
            self.X[indices, :], self.Y, self.Z
        )

    def evaluate(self, indices, is_permutation=True, observed_cmi=None):
        """Evaluate a surrogate using the cached fixed correlation block.

        Set ``is_permutation=False`` when donor indices can repeat.
        """
        indices = np.asarray(indices)
        if is_permutation:
            x_scaled = self.x_scaled[indices]
            corr_x = self.corr_x
        else:
            sampled = self.X[indices]
            std_x = np.std(sampled, axis=0, ddof=1)
            if np.any(std_x < 1e-12):
                return self._reference(indices)
            x_scaled = (sampled - np.mean(sampled, axis=0)) / std_x
            corr_x = (x_scaled.T @ x_scaled) / (self.n - 1)

        cross = (x_scaled.T @ self.yz_scaled) / (self.n - 1)
        corr = np.empty((self.kx + self.ky + self.kz,) * 2)
        corr[: self.kx, : self.kx] = corr_x
        corr[: self.kx, self.kx :] = cross
        corr[self.kx :, : self.kx] = cross.T
        corr[self.kx :, self.kx :] = self.corr_yz

        sign_xz, s_xz = np.linalg.slogdet(corr[np.ix_(self._xzidx, self._xzidx)])
        sign_xyz, s_xyz = np.linalg.slogdet(corr)
        if (
            sign_xz <= 0
            or sign_xyz <= 0
            or not np.isfinite(s_xz)
            or not np.isfinite(s_xyz)
            or min(s_xz, s_xyz) < -24
        ):
            return self._reference(indices)
        value = 0.5 * (s_xz + self._s_yz - self._s_z - s_xyz)
        # Use the reference calculation near the decision boundary.
        if observed_cmi is not None and abs(value - observed_cmi) <= 1e-10 * max(
            1.0, abs(observed_cmi)
        ):
            return self._reference(indices)
        return value if np.isfinite(value) else self._reference(indices)
