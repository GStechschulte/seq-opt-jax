from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from jax import Array
from jax.typing import ArrayLike
from scipy.optimize import linprog


class MPCInfo(NamedTuple):
    objective: Array


class State(NamedTuple):
    x: Array  # Battery SoC and electricity price


class MPCAlgorithm(NamedTuple):
    init: Callable
    step: Callable


class Decision(NamedTuple):
    u: Array  # Amount of energy to buy/sell to or from grid


class MPCParams(NamedTuple):
    T: int  # Length of simulation
    H: int  # Control horizon


class SystemParams(NamedTuple):
    R_max: float
    R_min: float
    alpha: float
    eta: float  # Battery discharge/charge efficiency


def objective(u: Decision):
    pass


def build_mpc(model: Callable, system_params: SystemParams, mpc_params: MPCParams):
    def init(init_state: Array) -> State:
        return State(x=init_state)

    def step(state: State):
        for k in range(mpc_params.H):
            pred_x = model(k)

    return MPCAlgorithm(init, step)


def main():
    prices_df = pl.read_excel(
        source="./data/energy_storage.xlsx", sheet_name="Raw Data"
    ).select(["Day", "PJM RT LMP"])

    sys_params = SystemParams(
        R_max=10.0,
        R_min=0.0,
        alpha=5.0,
        eta=0.9,
    )
    mpc_params = MPCParams(T=48, H=24)

    prices = prices_df.select("PJM RT LMP").to_numpy()[: mpc_params.T].flatten()

    def model(x):
        return prices[x : mpc_params.H + x]

    mpc = build_mpc(model, sys_params, mpc_params)

    # Initialize state
    x0 = np.array([[5.0, prices[0]]])
    state = mpc.init(x0)
    mpc.step(state)


if __name__ == "__main__":
    main()
