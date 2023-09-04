import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path

import tinygp
from tinygp.helpers import JAXArray, dataclass
from tinygp.kernels.stationary import Stationary
import jax
import jax.numpy as jnp

from transdimensional_sampler import Transdimensional_sampler

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

jax.config.update("jax_enable_x64", True)

df = pd.read_csv(Path(__file__).parents[1] / 'bao.csv', comment='#')

t_train = df['z'].values
y_train = df['H'].values
y_train_err = df['dH'].values


@dataclass
class Matern72(Stationary):
    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r = self.distance.distance(X1, X2) / self.scale
        arg = np.sqrt(7) * r
        return (1 + arg + 2 * (arg ** 2) / 5 + (arg ** 3) / 15) * jnp.exp(-arg)


@dataclass
class Linear(tinygp.kernels.Kernel):
    sigma_b: JAXArray
    sigma_v: JAXArray

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.sigma_b ** 2 + self.sigma_v ** 2 * X1 * X2


def k1(theta):
    return (theta[0] ** 2) * tinygp.kernels.Matern32(theta[1], tinygp.kernels.distance.L2Distance())


def k2(theta):
    return (theta[0] ** 2) * tinygp.kernels.Matern52(theta[1], tinygp.kernels.distance.L2Distance())


def k3(theta):
    return (theta[0] ** 2) * Matern72(theta[1], tinygp.kernels.distance.L2Distance())


def k9(theta):
    return Linear(theta[0], theta[1])


def noise(theta):
    return tinygp.noise.Diagonal(jnp.full_like(y_train, theta ** 2))


# Transdimensional sampler with 4 kernels
A_range = 0., 500.
l_range = 0., 20.
sampler = Transdimensional_sampler(
    t_train,
    y_train,
    [k1, k2, k3, k9],
    [2, 2, 2, 2],
    [[A_range, l_range], [A_range, l_range], [A_range, l_range],
     [A_range, A_range]],
    ['$A_\mathrm{M32}$', '$l_\mathrm{M32}$',
     '$A_\mathrm{M52}$', '$l_\mathrm{M52}$',
     '$A_\mathrm{M72}$', '$l_\mathrm{M72}$',
     '$\sigma_b$', '$\sigma_v$'],
    lambda theta: theta[0],
    1,
    [(-900, 1000)],
    ['$m$'],
    noise,
    1,
    [A_range],
    [r'$\sigma$']
)

# Run transdimensional sampler using nested sampling
sampler.run_polychord()

# Evidence
print(sampler.log_evidence())
# Kernel evidences
logZi, sigma_logZi = sampler.log_kernel_evidences()
print(logZi, sigma_logZi)
# Kernel probabilities
print(sampler.p(logZi, sigma_logZi))

# Corner plot of M32 kernel hyperparameters
# Zeroth kernel is the M32 kernel
samples = sampler.read_samples([0])
# M32 hyperparameters have indices 3 and 4 in the transdimensional vector
corner_fig, corner_ax = samples.plot_2d([3, 4])
corner_fig.savefig('corner.png')

# Plot kernel-marginalised predictive distribution
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
ax.errorbar(t_train, y_train, y_train_err, fmt='.')
sampler.plot_predictive_curve(fig, ax, np.linspace(-1, 3, 100), 'combine', None, 20)
ax.set_xlabel('z')
ax.set_ylabel('H')
ax.set_title('Kernel-marginalised GP predictive distribution')
fig.savefig('plot.png')
