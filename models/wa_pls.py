import numpy as np

class WAPLS:
    def __init__(self, n_components=2, weighted=False, standardize=True, lean=False):
        self.n_components = n_components
        self.weighted = weighted
        self.standardize = standardize
        self.lean = lean

    def _to_float_array(self, arr, name):
        arr = np.asarray(arr)
        if arr.dtype == object:
            try:
                arr = arr.astype(float)
            except Exception as e:
                raise ValueError(f"{name} must be numeric. Problem: {e}")
        return np.array(arr, dtype=float)

    def fit(self, X, Y):
        X = self._to_float_array(X, "X")
        Y = self._to_float_array(Y, "Y").reshape(-1, 1)
        nr, nc = X.shape
        nPLS = self.n_components

        spec_count = np.count_nonzero(X, axis=0)

        # Weighting and standardization
        if not self.weighted:
            self.meanY = float(Y.mean())
            Yc = Y - self.meanY
            R = np.ones((nr, 1))
            if self.standardize:
                C = X.std(axis=0, ddof=1).reshape(-1, 1)
                C[C < 1e-12] = 1.0
            else:
                C = np.ones((nc, 1))
        else:
            C = X.sum(axis=0).reshape(-1, 1)
            C[C < 1e-12] = 1.0
            R = X.sum(axis=1).reshape(-1, 1)
            Ytot = R.sum()
            wtr = R / Ytot
            self.meanY = float((Y.T @ wtr).item())
            Yc = Y - self.meanY

        # Initial gradient
        if not self.weighted:
            g = X.T @ Yc / C
            gamma0 = np.sum(g ** 2)
        else:
            Wc = C / C.sum()
            g = X.T @ Yc / C
            gamma0 = np.sum(g ** 2 * Wc)

        d = g.copy()
        b = np.zeros((nc, 1))
        T_list, P_list, meanT_list = [], [], []

        for i in range(nPLS):
            if not self.weighted:
                t = X @ (d / C)
                if T_list:
                    Tmat = np.column_stack(T_list)
                    t -= Tmat @ (Tmat.T @ t / np.sum(Tmat ** 2, axis=0))
                meant = t.mean()
                t -= meant
                tau = np.sum(t ** 2)
            else:
                t = (X @ d) / R
                Wr = R / R.sum()
                tau = np.sum((t ** 2) * Wr)

            if tau > 1e-12:
                alpha = gamma0 / tau
                b += d * alpha
                Yc -= t * alpha

                # Update gradient
                if not self.weighted:
                    g = X.T @ Yc / C
                    gamma = np.sum(g ** 2)
                else:
                    g = X.T @ Yc / C
                    gamma = np.sum(g ** 2 * Wc)

                d = g + (gamma / gamma0) * d
                gamma0 = gamma

                T_list.append(t.flatten())
                P_list.append(d.flatten())
                if not self.weighted:
                    meanT_list.append(meant * alpha)
            else:
                break

        self.Beta = b.flatten()
        self.T = np.column_stack(T_list) if T_list else np.empty((nr, 0))
        self.P = np.column_stack(P_list) if P_list else np.empty((nc, 0))
        self.meanT = np.array(meanT_list) if meanT_list else np.zeros(self.P.shape[1])
        self.sdX = C.flatten() if not self.weighted else np.zeros(nc)
        return self

    def predict(self, X):
        X = self._to_float_array(X, "X")
        nr, nc = X.shape
        Beta = self.Beta.reshape(-1, 1)
        est = X @ Beta

        if self.weighted:
            row_sums = X.sum(axis=1).reshape(-1, 1)
            row_sums[row_sums < 1e-12] = 1.0
            est /= row_sums
            est += self.meanY  # <— Add this line
        else:
            est += self.meanY
            if self.meanT is not None and len(self.meanT) > 0:
                est -= np.cumsum(self.meanT)[-1]

        return est
