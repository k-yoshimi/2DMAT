import numpy as np

import py2dmat
import py2dmat.solver.function

class Solver(py2dmat.solver.function.Solver):
    """
    A solver for calculating and optimizing electrical conductivity tensors.

    This solver implements a model for calculating the electrical conductivity
    tensor components (σxx and σxy) based on carrier properties and magnetic field
    strength. It's designed to fit experimental data by optimizing carrier
    concentrations and mobilities.

    The model considers both electron and hole carriers, where:
    - σxx represents the longitudinal conductivity
    - σxy represents the Hall conductivity
    - B represents the magnetic field strength

    Attributes
    ----------
    x : np.ndarray
        Current parameter values being evaluated
    fx : float
        Current objective function value
    B_target : np.ndarray
        Target magnetic field values from experimental data
    sigma_experiment : np.ndarray
        Experimental conductivity tensor data
    alpha : float
        Weight parameter for balancing σxx and σxy contributions in the objective function
    sigma_experiment_mean_0 : float
        Mean value of experimental σxx data, used for normalization
    sigma_experiment_mean_1 : float
        Mean value of experimental σxy data, used for normalization
    """

    x: np.ndarray
    fx: float

    def __init__(self, info: py2dmat.Info) -> None:
        """
        Initialize the sigma solver with experimental data and parameters.

        Parameters
        ----------
        info : py2dmat.Info
            Configuration object containing solver parameters and data paths.
            Expected keys:
            - solver.path_to_reference: Path to experimental data file
            - algorithm.param.alpha: Weight parameter (default: 0.5)
        """
        super().__init__(info)
        self._name = "sigma"
        # Get target data from CSV file
        info_s = info.solver
        _path_to_target_data = info_s.get("path_to_reference", "30Kexperimentdata.csv")
        data = np.genfromtxt(_path_to_target_data, delimiter=',', encoding="utf-8-sig")[1:,:3]
        self.B_target = data[:,0]  # Magnetic field values
        self.sigma_experiment = data[:,1:]  # Experimental conductivity data
        # Normalize experimental data by first value
        self.sigma_experiment /= self.sigma_experiment[0][0]
        self._func = self._sigma_diff
        print(info.algorithm)
        # Get alpha parameter for weighting σxx vs σxy contributions
        self.alpha = info.algorithm["param"].get("alpha", 0.5)
        # Pre-calculate mean values for normalization to improve performance
        self.sigma_experiment_mean_0 = np.mean(self.sigma_experiment[:,0])
        self.sigma_experiment_mean_1 = np.mean(self.sigma_experiment[:,1])

    def _sigma(self, B_target, xs):
        """
        Calculate conductivity tensor components (σxx, σxy) using carrier concentrations.

        This method calculates the conductivity components using the carrier
        concentrations directly, without considering the charge.

        Parameters
        ----------
        B_target : np.ndarray
            Array of magnetic field values
        xs : np.ndarray
            Parameter array containing:
            - First half: carrier concentrations (s_i)
            - Second half: carrier mobilities (μ_i)

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Two arrays containing:
            - σxx: Longitudinal conductivity values
            - σxy: Hall conductivity values
        """
        N = int(len(xs) / 2)
        xs = np.array(xs)
        s = xs[:N]  # Carrier concentrations
        myu = xs[N:]  # Carrier mobilities
        B_target = B_target.reshape(-1, 1)

        # Calculate σxx (longitudinal conductivity)
        sigma_xx = np.sum(s / (1 + myu ** 2 * B_target ** 2), axis=1)
        
        # Calculate σxy (Hall conductivity)
        # Split calculation for electrons (negative) and holes (positive)
        sigma_xy_1 = (-s[:int(N / 2)] * myu[:int(N / 2)] / (1 + myu[:int(N / 2)] ** 2 * B_target ** 2)
                      + s[int(N / 2):] * myu[int(N / 2):] / (1 + myu[int(N / 2):] ** 2 * B_target ** 2)) * B_target
        sigma_xy = np.sum(sigma_xy_1, axis=1)
        return sigma_xx, sigma_xy

    def _sigma_n_mu(self, B_target, xs):
        """
        Calculate conductivity tensor components using carrier density and mobility.

        This method calculates the conductivity components using the carrier density
        (n) and mobility (μ) directly, considering the charge (e) in the calculation.
        This is an alternative implementation that explicitly separates carrier
        density and mobility parameters.

        Parameters
        ----------
        B_target : np.ndarray
            Array of magnetic field values
        xs : np.ndarray
            Parameter array containing:
            - First half: carrier densities (n_i)
            - Second half: carrier mobilities (μ_i)

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Two arrays containing:
            - σxx: Longitudinal conductivity values
            - σxy: Hall conductivity values

        Notes
        -----
        The conductivity components are calculated as:
        σxx(B) = Σ(e * n_i * μ_i / (1 + μ_i²B²))
        σxy(B) = Σ(sign(n_i) * e * n_i * μ_i² * B / (1 + μ_i²B²))
        where:
        - n_i = carrier density
        - μ_i = carrier mobility
        - e = elementary charge
        - B = magnetic field strength
        - sign(n_i) = -1 for electrons, +1 for holes
        """
        N = int(len(xs) / 2)
        xs = np.array(xs)
        en = xs[:N]  # Carrier densities
        myu = xs[N:]  # Carrier mobilities
        B_target = B_target.reshape(-1, 1)

        # Calculate σxx (longitudinal conductivity)
        # Note: en already includes the charge factor
        sigma_xx = np.sum(en * myu / (1 + myu ** 2 * B_target ** 2), axis=1)
        
        # Calculate σxy (Hall conductivity)
        # Split calculation for electrons (negative) and holes (positive)
        # Note the μ² term in the numerator for σxy
        sigma_xy_1 = (-en[:int(N / 2)] * myu[:int(N / 2)] ** 2 / (1 + myu[:int(N / 2)] ** 2 * B_target ** 2)
                      + en[int(N / 2):] * myu[int(N / 2):] ** 2 / (1 + myu[int(N / 2):] ** 2 * B_target ** 2)) * B_target
        sigma_xy = np.sum(sigma_xy_1, axis=1)
        return sigma_xx, sigma_xy

    def _sigma_diff(self, xs: np.ndarray) -> float:
        """
        Calculate the objective function value for optimization.

        This function computes the weighted sum of normalized mean squared errors
        between calculated and experimental conductivity components. The errors
        are normalized by the mean values of the experimental data to ensure
        equal weighting of relative errors.

        Parameters
        ----------
        xs : np.ndarray
            Parameter array containing carrier concentrations and mobilities

        Returns
        -------
        float
            Objective function value combining σxx and σxy errors:
            α * δσxx + (1-α) * δσxy
            where δσxx and δσxy are normalized root mean squared errors
        """
        # Calculate conductivity components using carrier density and mobility
        sigma_xx, sigma_xy = self._sigma_n_mu(self.B_target, xs)

        # Calculate normalized root mean squared errors using pre-computed means
        delta_sigma_xx = np.sqrt(np.mean(((self.sigma_experiment[:,0] - sigma_xx)/self.sigma_experiment_mean_0)**2))
        delta_sigma_xy = np.sqrt(np.mean(((self.sigma_experiment[:,1] - sigma_xy)/self.sigma_experiment_mean_1)**2))
        
        # Combine errors with weight parameter alpha
        alpha = self.alpha
        return alpha*delta_sigma_xx + (1.0-alpha)*delta_sigma_xy
