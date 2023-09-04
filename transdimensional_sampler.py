from enum import Enum, auto
from functools import partial
from typing import Callable, Union, Optional

import fgivenx
from pypolychord import settings
from pypolychord.priors import UniformPrior
import pypolychord as pc
import anesthetic

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

import tinygp
from tinygp.helpers import JAXArray
import jax


class Transdimensional_sampler():
    """
    Implements the transdimensional sampler:
        - Handles array index manipulations
        - Constructs the prior and loglikelihood functions
        - Calls PolyChord nested sampling run.
        - Handles sample post-processing and plotting with the anesthetic library.
    """

    # Define 'instantiation_func' type which is a function which takes hyperparameters as input and returns the kernel/mean function/noise term
    instantiation_func = Callable[[JAXArray], Union[tinygp.kernels.Kernel, Union[Callable, JAXArray], tinygp.noise.Noise]]
    # Define type which saves the lower and upper limit of the uniform prior bound.
    parameter_bound = tuple[float, float]

    def __init__(
            self,
            # Dataset
            X_train,
            y_train,
            # Kernel
            kernel_instantiation_funcs: list[instantiation_func],
            num_kernel_parameters: list[int],
            kernel_parameter_bounds: list[list[parameter_bound]],
            kernel_names: list[str],
            # Mean function
            mean_instantiation_func: instantiation_func,
            num_mean_parameters: int,
            mean_parameter_bounds: list[parameter_bound],
            mean_parameter_names: list[str],
            # Noise
            noise_instantiation_func: instantiation_func,
            num_noise_parameters: int,
            noise_parameter_bounds: list[parameter_bound],
            noise_parameter_names: list[str],
            # Custom prior functions
            prior: Optional[list[Callable]] = None
    ):
        """
        Constructor saves object attributes, computes slice objects and prepares the prior objects.
        The ordering of the individual hyperparameters in the transdimensional vector \Phi is 
        (kernel choice, mean function hyperparameters, noise hyperparameters, kernel hyperparameters for all kernels)
        """

        self.X_train = X_train
        self.y_train = y_train

        # For each kernel, save the 'instantiation_func'. This is called to create a 'tinygp.Kernel' object.
        self.kernel_instantiation_funcs = kernel_instantiation_funcs
        # Save the number of kernels to sample from.
        self.num_kernels = len(self.kernel_instantiation_funcs)
        # For each kernel, save the number of kernel hyperparameters.
        self.num_kernel_parameters = num_kernel_parameters
        # For each kernel, save the uniform prior bounds for each hyperparameter.
        self.kernel_parameter_bounds = kernel_parameter_bounds
        # Save names of the kernels as strings for plotting purposes.
        self.kernel_names = kernel_names

        # Similar, but for the mean function.
        self.mean_instantiation_func = mean_instantiation_func
        self.num_mean_parameters = num_mean_parameters
        self.mean_parameter_bounds = mean_parameter_bounds
        self.mean_parameter_names = mean_parameter_names

        # Similar, but for the noise term.
        self.noise_instantiation_func = noise_instantiation_func
        self.num_noise_parameters = num_noise_parameters
        self.noise_parameter_bounds = noise_parameter_bounds
        self.noise_parameter_names = noise_parameter_names

        # Calculate the dimension of the total hyperparameter vector \Phi: 
        # (kernel choice hyperparameter) + (kernel hyperparameters) + (mean hyperparameters) + (noise hyperparameters)
        self.n_dims = 1 + np.sum(self.num_kernel_parameters) + self.num_mean_parameters + self.num_noise_parameters

        # Indices to slice the parameters for each kernel from the total parameter vector.
        # These are collected as 'slice' objects in the list 'indices_to_slice_kernel_params'.
        self.indices_to_slice_kernel_params = []
        lower = 1 + self.num_mean_parameters + self.num_noise_parameters
        for step in self.num_kernel_parameters:
            upper = lower + step
            self.indices_to_slice_kernel_params.append(slice(lower, upper))
            lower = upper
        assert lower == self.n_dims
        assert len(self.indices_to_slice_kernel_params) == self.num_kernels

        # 'slice' objects for the mean function and noise term.
        self.indices_to_slice_mean_params = slice(1, 1 + self.num_mean_parameters)
        self.indices_to_slice_noise_params = slice(
            1 + self.num_mean_parameters, 1 + self.num_mean_parameters + self.num_noise_parameters
        )

        # List of priors in terms of inverse transforms for PolyChord. For the categorical variable, the interval [0, 1] is multiplied by the number of categories and rounded down to an integer.
        if prior is None:
            self.prior_funcs = [lambda x: np.floor(self.num_kernels * x)] \
                               + [UniformPrior(*bounds) for bounds in self.mean_parameter_bounds] \
                               + [UniformPrior(*bounds) for bounds in self.noise_parameter_bounds] \
                               + [UniformPrior(*b) for bs in self.kernel_parameter_bounds for b in bs]
        else:
            # If custom priors are provided, use these instead of the uniform priors.
            self.prior_funcs = prior

    def prior(self, unit_cube):
        """
        Prior transformation from the unit cube. Passed to PolyChord in the method 'run_polychord'.
        Args:
            unit_cube: 1D array of the length of the number of dimensions of \Phi.
        Returns:
            \Phi as a 1D array, computed from the prior transform.
        """

        return np.array([f(x) for f, x in zip(self.prior_funcs, unit_cube)])

    def build_gp(self, kernel_choice, mean, noise_params, kernel) -> tinygp.GaussianProcess:
        """
        Instantiate GaussianProcess object with given hyperparameters.
        Args:
            - kernel_choice: the categorical variable encoded as an integer to choose the kernel.
            - mean: Mean function
            - noise_params: Array of noise hyperparameters.
            - kernel: tinygp.Kernel object.
        """

        # Instantiate the tinygp.Noise object from the noise hyperparameters.
        noise = self.noise_instantiation_func(noise_params)
        # Return the tinygp.GaussianProcess object.
        return tinygp.GaussianProcess(kernel, self.X_train, noise=noise, mean=mean)

    # jit-compile this function. Exclude the 'self' argument from the jit compilation to avoid re-compilation when the object attributes change.
    @partial(jax.jit, static_argnums=(0,))
    def single_gp_loglikelihood(self, kernel_choice, mean, noise_params, kernel: tinygp.kernels.Kernel):
        """
        Function which calculates the single-kernel loglikelihood for given hyperparameters. jit-compiled for speed.
        Args:
            - kernel_choice: the categorical variable encoded as an integer to choose the kernel.
            - mean: Mean function.
            - noise_params: Array of noise hyperparameters.
            - kernel: tinygp.Kernel object.
        Returns:
            Single-kernel loglikelihood evaluated at the hyperparameters
        """

        # Note: Kernel class can be passed as a dynamic argument (not possible with ordinary numpy arrays) so can use jax.jit.
        gp = self.build_gp(kernel_choice, mean, noise_params, kernel)
        return gp.log_probability(self.y_train)

    def theta_to_gp_params(self, phi):
        """
        Split the transdimensional hyperparameter vector phi into the hyperparameters for a given kernel choice.
        Args:
            phi: Transdimensional hyperparameter vector \Phi
        Returns: 
            - kernel_choice: the categorical variable encoded as an integer to choose the kernel.
            - mean: Mean function.
            - noise_params: Array of noise hyperparameters.
            - kernel: tinygp.Kernel object.
        """

        # Have to do dynamic array slicing here (i.e. array slicing which would produce an array with a length which depends on the input values of this function at runtime) outside the jitted function.
        # This is because jit-compilation only works with fixed-length arrays.

        # Extract categotical hyperparameter.
        kernel_choice = phi[0].astype(int)
        # Extract mean function hyperparameters.
        mean_params = phi[self.indices_to_slice_mean_params]
        # Instantiate the mean function.
        mean = self.mean_instantiation_func(mean_params)

        # Extract the noise hyperparameters.
        noise_params = phi[self.indices_to_slice_noise_params]

        # Extract the kernel hyperparameters, accounting for the kernel choice.
        kernel_params = phi[self.indices_to_slice_kernel_params[kernel_choice]]
        # Instantiate the tinygp.Kernel object.
        kernel = self.kernel_instantiation_funcs[kernel_choice](kernel_params)

        return kernel_choice, mean, noise_params, kernel

    def loglikelihood(self, phi: np.ndarray) -> np.float64:
        """
        Calculates the loglikelihood from the transdimensional hyperparameter vector phi.
        Args:
            phi: Transdimensional hyperparameter vector \Phi
        Returns:
            Loglikelihood evaluated at phi.
        """

        # Split phi into single-kernel hyperparameters and pass these to the single-kernel loglikelihood. Then convert the type 'DeviceArray to a numpy 'float'.
        return np.float64(self.single_gp_loglikelihood(*self.theta_to_gp_params(phi)))

    def run_polychord(self, nlive: Optional[int] = None, file_root: Optional[str] = 'samples') -> None:
        """
        Pass the loglikelihood and prior to PolyChord and run PolyChord.
        Args:
            - nlive: Number of live points.
            - file_root: Root directory for PolyChord raw output.
        """

        # Prepare PolyChord settings object.
        base_dir = str(Path(__file__).parent / 'chains')
        pc_settings = settings.PolyChordSettings(
            nDims=self.n_dims, nDerived=0, base_dir=base_dir, file_root=file_root, seed=1
        )
        # Maximum output to console.
        pc_settings.feedback = 3
        # Write and read from a 'resume' file.
        pc_settings.read_resume = True
        pc_settings.write_resume = True
        # Scale number of live points linearly with number of kernels.
        pc_settings.nlive = self.num_kernels * pc_settings.nlive
        # If nlive is provided, use this instead.
        if nlive is not None:
            pc_settings.nlive = nlive

        # Wrap loglikelihood function, as required by PolyChord.
        def loglikelihood_wrapper(phi):
            return self.loglikelihood(phi), []

        # Start sampling with PolyChord.
        pc.run_polychord(loglikelihood_wrapper, prior=self.prior, nDims=self.n_dims, nDerived=0, settings=pc_settings)

    def read_samples(self, kernel_choice: Optional[list[int]] = None, file_root: Optional[str] = 'samples'):
        """
        Use anesthetic to read raw PolyChord output.
        Args:
            - kernel_choice: List of kernels to take samples from. 'None' corresponds to all kernels.
            - file_root: Root of the PolyChord raw output.
        Returns:
            Weighted DataFrame containing posterior samples of the kernels specified by the 'kernel_choice'.
        """

        base_dir = 'chains'
        ns = anesthetic.read_chains(root=str(Path(__file__).parent / base_dir / file_root))

        # Give each parameter its label, for plotting purposes.
        indices = np.arange(self.n_dims)
        # Concatenate label lists.
        labels = ['Kernel'] + self.mean_parameter_names + self.noise_parameter_names + self.kernel_names
        assert len(indices) == len(labels)
        for i, l in zip(indices, labels):
            ns.set_label(i, l)

        if kernel_choice is None:
            return ns
        # Take only the samples corresponding to the specified kernels, i.e. the first entry in the PolyChord vector must match one of the integers specified in kernel_choice. Then, recompute.
        return ns.loc[ns[0].isin(kernel_choice)].recompute()

    def log_evidence(self) -> tuple[float, float]:
        """
        Compute lnZ and its error. Work in log-space because the evidence is normally distributed in log-space so error bars are symmetric.
        Returns:
            - mean: Log of the evidence of the joint posterior of the kernels and kernel hyperparameters.
            - std: Standard deviation of the evidence.
        """

        ns = self.read_samples()
        # Calculate mean and standard deviation from 1000 logZ samples.
        logZ_draws = ns.logZ(nsamples=1000).values
        mean = np.mean(logZ_draws)
        std = np.std(logZ_draws)
        return float(mean), float(std)

    def log_kernel_evidences(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the evidence for the posterior distribution over kernel choices.
        Returns:
            - log_Z: Array of ln(evidence) for kernel_0, kernel_1, ..., kernel_(num_kernels)
            - errors: Corresponding array of standard deviations for each value in logEvidences
        """

        log_Z = np.zeros(self.num_kernels)
        errors = np.zeros(self.num_kernels)

        # For each kernel, calculate the evidence and its error from 1000 samples.
        for i in range(self.num_kernels):
            ns = self.read_samples([i])
            logZ_draws = ns.logZ(nsamples=1000)
            log_Z[i] = logZ_draws.mean()
            errors[i] = logZ_draws.std()

        return log_Z, errors

    def p(self, logZ, sigma_logZ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the kernel posterior probabilities, p_k, and the standard deviation on each using the evidence, lnZ_k.
        Args:
            - logZ: 1D numpy array containing lnZ_k for each kernel
            - sigma_logZ: 1D numpy array containing the standard deviations corresponding to logZ
        Returns:
            - pr: 1D numpy array containing p_k for each kernel
            - sigma_pr 1D numpy array containing the standard deviations corresponding to pr
        """

        # Dimensions: (N) -> (1, N)
        logZ = np.atleast_2d(logZ)
        sigma_logZ = np.atleast_2d(sigma_logZ)

        # Dimensions: (N, 1) x (1, N) = (N, N) with indices (j, i)
        exp_diff = np.exp(logZ.T - logZ)
        # Dimensions: (N, N) -> (N) with index i
        pr = 1. / np.sum(exp_diff, axis=0)

        # Dimensions: (N, 1) x (N, N) = (N, N) with indices (j, i)
        second_term = (sigma_logZ.T * exp_diff) ** 2
        np.fill_diagonal(second_term, val=0)
        # Dimensions: [(N), index i] x [(N), index i] = (N), index i
        second_term = pr ** 2 * np.sum(second_term, axis=0)

        # Dimensions: (N) x (N) x (N) = (N)
        sigma_pr = pr * np.sqrt(sigma_logZ[0, :] ** 2 + second_term)

        return pr, sigma_pr

    def plot_predictive_curve(
            self,
            fig: plt.Figure,
            ax: plt.Axes,
            X_test,
            mode: str,
            kernel_choice: Optional[list[int]] = None,
            n_post = 20,
            label='NS',
            color='red',
            colormap='Reds'
    ) -> None:
        """
        Plot the predictions on the test inputs 'X_test' in the plt.Axes object 'ax'.
        Args:
            - fig: Figure object, corresponding to ax.
            - ax: Axes object to be written to.
            - X_test: Test inputs.
            - kernel_choice: Either string 'all' (predictions from all kernels are combined) or a list of ints, specifying
                the kernels whose predictions should be plotted.
            - mode: Either 'naive' (n_post samples are overlayed on the plot), 'fgivenx_lines'/'fgivenx_contour'
                (fgivenx package is used), or 'combine' (mean and variance of the Gaussian mixture distribution are
                plotted by combining curves with analytical formulae)
            - n_post: Number of posterior samples to take
            - color: Color of the plotted functions
            - colormap: Colormap for the 'fgivenx_contour' mode.
            - label: Label for the plotted curves
        """
        class Modes(Enum):
            naive = auto()
            fgivenx_lines = auto()
            fgivenx_contour = auto()
            combine = auto()

        try:
            mode = Modes[mode]
        except KeyError as e:
            raise e

        ns = self.read_samples(kernel_choice)

        # Sample n_post points.
        post_points = ns.sample(n_post, replace=True)
        post_points = post_points.to_numpy()[..., :-3]

        def conditioned_gp(theta) -> tinygp.GaussianProcess:
            """ Get the GP conditioned on the parameters 'theta', i.e. the predictive distribution. """
            kernel_choice, mean, noise_params, kernel = self.theta_to_gp_params(theta)
            # Need to instantiate noise object here again because it must be passed to gp.condition() explicitly.
            noise = self.noise_instantiation_func(noise_params)
            _, gp = self.build_gp(kernel_choice, mean, noise_params, kernel) \
                .condition(self.y_train, X_test, kernel=kernel)
            return gp

        if mode is Modes.naive:
            for i, theta in enumerate(post_points):
                gp = conditioned_gp(theta)
                mean, std = gp.loc, np.sqrt(gp.variance)
                # Plot curve. Only use a label in the first loop to avoid duplicate labels.
                ax.plot(X_test, mean, color=color, label=label if i == 0 else None)
                ax.fill_between(X_test, mean - std, mean + std, alpha=.05, color=color, linewidth=0)

        elif mode in [Modes.fgivenx_lines, Modes.fgivenx_contour]:
            # Sample from mixture distribution:
            # 1. Pick random GP
            # 2. Sample from that GP
            sample_curves = np.zeros(shape=(n_post, len(X_test)))
            jax_key = jax.random.PRNGKey(1)
            for i, theta in enumerate(post_points):
                if i % 1 == 0: print(i)
                # Boilerplate code to use jax RNG
                jax_key, jax_subkey = jax.random.split(jax_key)
                gp = conditioned_gp(theta)
                sample_curves[i, :] = gp.sample(jax_subkey, shape=(1,))

            # Create cache directory for fgivenx package
            cache = Path(__file__).parent / 'fgivenx_plot_cache'
            cache.mkdir(parents=True, exist_ok=True)
            cache = str(cache) + '/'

            if mode is Modes.fgivenx_lines:
                fgivenx.plot_lines(
                    # The sample curves have already been calculated, so pass a fake function which just returns the curves
                    lambda x, theta: theta,
                    X_test,
                    sample_curves,
                    ax,
                    color=color,
                    cache=cache
                )
            elif mode is Modes.fgivenx_contour:
                cbar = fgivenx.plot_contours(
                    # The sample curves have already been calculated, so pass a fake function which just returns the curves
                    lambda x, theta: theta,
                    X_test,
                    sample_curves,
                    ax,
                    colors=mpl.colormaps[colormap],
                    cache=cache
                )
                fig.colorbar(cbar)

        elif mode is Modes.combine:
            mean_curves = np.zeros(shape=(n_post, len(X_test)))
            std_curves = np.zeros(shape=(n_post, len(X_test)))
            for i, theta in enumerate(post_points):
                gp = conditioned_gp(theta)
                mean_curves[i, :], std_curves[i, :] = gp.loc, np.sqrt(gp.variance)
            # Combine predictions from posterior samples
            mean = np.mean(mean_curves, axis=0)
            std = np.sqrt(np.mean(std_curves ** 2, axis=0) + np.var(mean_curves, axis=0))
            ax.plot(X_test, mean, color=color, label=label)
            ax.fill_between(X_test, mean - std, mean + std, alpha=.2, color=color, linewidth=0)