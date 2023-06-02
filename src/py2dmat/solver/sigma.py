import numpy as np

import py2dmat
import py2dmat.solver.function

class Solver(py2dmat.solver.function.Solver):
    """Function Solver with pre-defined benchmark functions"""

    x: np.ndarray
    fx: float

    def __init__(self, info: py2dmat.Info) -> None:
        """
        Initialize the solver.

        Parameters
        ----------
        info: Info
        """
        super().__init__(info)
        self._name = "sigma"
        #Get target data
        info_s = info.solver
        _path_to_target_data = info_s.get("path_to_reference", "30Kexperimentdata.csv")
        data = np.genfromtxt(_path_to_target_data, delimiter=',', encoding="utf-8-sig")[1:,:3]
        self.B_target = data[:,0]
        self.sigma_experiment = data[:,1:]
        self.sigma_experiment /= self.sigma_experiment[0][0]
        self._func = self._sigma_diff
        print(info.algorithm)
        self.alpha = info.algorithm["param"].get("alpha", 0.5)
        self.sigma_experiment_mean_0 = np.mean(self.sigma_experiment[:,0])
        self.sigma_experiment_mean_1 = np.mean(self.sigma_experiment[:,1])
        
    def _sigma(self, B_target, xs):
        """
        xs : n_1, n_2, ..., n_N, mu_1, mu_2, ..., mu_N
        sigma_xx (B) = \sum_{i=1}^N e n_i \mu_i / (1+\mu_i^2 B^2)
                     = \sum_{i=1}^N s_i / (1+\mu_i^2 B^2)
        sigma_xy (B) = \sum_{i=1}^N sign(n_i) e n_i \mu_i^2 B/ (1+\mu_i^2 B^2)
                     = \sum_{i=1}^N sign(s_i) s_i \mu_i B/ (1+\mu_i^2 B^2)
        Here, sign(s_i) = -1 if i <= N/2 (electron), 1 if i > N/2 (hole)
        """
        N = int(len(xs) / 2)
        xs = np.array(xs)
        s = xs[:N]
        myu = xs[N:]
        B_target = B_target.reshape(-1, 1)

        # sigma_xxの計算
        sigma_xx = np.sum(s / (1 + myu ** 2 * B_target ** 2), axis=1)
        # sigma_xyの計算
        sigma_xy_1 = (-s[:int(N / 2)] * myu[:int(N / 2)] / (1 + myu[:int(N / 2)] ** 2 * B_target ** 2)
                      + s[int(N / 2):] * myu[int(N / 2):] / (1 + myu[int(N / 2):] ** 2 * B_target ** 2)) * B_target
        sigma_xy = np.sum(sigma_xy_1, axis=1)
        return sigma_xx, sigma_xy

    def _sigma_n_mu(self, B_target, xs):
        """
        xs : n_1, n_2, ..., n_N, mu_1, mu_2, ..., mu_N
        sigma_xx (B) = \sum_{i=1}^N e n_i \mu_i / (1+\mu_i^2 B^2)
        sigma_xy (B) = \sum_{i=1}^N sign(n_i) e n_i \mu_i^2 B/ (1+\mu_i^2 B^2)
        Here, sign(s_i) = -1 if i <= N/2 (electron), 1 if i > N/2 (hole)
        """
        N = int(len(xs) / 2)
        xs = np.array(xs)
        en = xs[:N]
        myu = xs[N:]
        B_target = B_target.reshape(-1, 1)

        # sigma_xxの計算
        sigma_xx = np.sum(en * myu / (1 + myu ** 2 * B_target ** 2), axis=1)
        # sigma_xyの計算
        sigma_xy_1 = (-en[:int(N / 2)] * myu[:int(N / 2)] ** 2 / (1 + myu[:int(N / 2)] ** 2 * B_target ** 2)
                      + en[int(N / 2):] * myu[int(N / 2):] ** 2 / (1 + myu[int(N / 2):] ** 2 * B_target ** 2)) * B_target
        sigma_xy = np.sum(sigma_xy_1, axis=1)
        return sigma_xx, sigma_xy

    def _sigma_diff(self, xs: np.ndarray) -> float:
        #sigma_xx, sigma_xy = self._sigma(self.B_target, xs)
        sigma_xx, sigma_xy = self._sigma_n_mu(self.B_target, xs)

        delta_sigma_xx = np.sqrt(np.mean(((self.sigma_experiment[:,0] - sigma_xx)/self.sigma_experiment_mean_0)**2))
        delta_sigma_xy = np.sqrt(np.mean(((self.sigma_experiment[:,1] - sigma_xy)/self.sigma_experiment_mean_1)**2))
        alpha = self.alpha
        return alpha*delta_sigma_xx + (1.0-alpha)*delta_sigma_xy
