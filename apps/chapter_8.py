import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # 8 - Energy Storage

    There is increased interest in using batteries to take advantage of price spikes, buying power when it is cheap, and selling it back when prices are high. Exploiting the variability in power prices on the grid to buy when prices are low and sell when they are high is known as *battery arbitrage*.
    """
    )
    return


@app.cell
def _():
    import time
    from abc import ABC, abstractmethod
    from dataclasses import dataclass
    from enum import IntEnum
    from functools import partial
    from typing import Any, NamedTuple, Callable, Tuple, Dict, Optional, List

    import numpyro
    numpyro.set_host_device_count(10)

    import jax
    jax.config.update("jax_num_cpu_devices", 10)

    import jax.numpy as jnp
    import numpyro.distributions as dist
    from jax import random, jit, vmap, grad
    from jax.scipy.optimize import minimize as jax_minimize
    from jax import Array
    from jax.typing import ArrayLike
    import matplotlib.pyplot as plt
    from numpyro.contrib.control_flow import scan
    from numpyro.infer import MCMC, NUTS, Predictive
    import polars as pl
    import seaborn as sns
    from scipy.optimize import minimize
    return (
        ABC,
        Array,
        MCMC,
        NUTS,
        NamedTuple,
        Predictive,
        Tuple,
        abstractmethod,
        dataclass,
        dist,
        jax,
        jnp,
        numpyro,
        pl,
        plt,
        random,
        scan,
        sns,
        time,
        vmap,
    )


@app.cell
def _(plt):
    plt.style.use("./styles.mplstyle")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 8.2 - Basic model

    We will develop an initial system of using a battery to buy from and sell to the grid to take advantage of price volatility.
    """
    )
    return


@app.cell
def _(NamedTuple):
    class State(NamedTuple):
        R_t: float # Amount of energy stored in the battery at time t
        p_t: float # Price of energy on the grid at time t
        eta: float = 0.9 # Battery conversion loss factor

    class Decision(NamedTuple):
        buy: float
        sell: float
        hold: float

    class Exogenous(NamedTuple):
        next_price: float

    class Policy(NamedTuple):
        theta_buy: float 
        theta_sell: float
        buy_amount: float = 1.0 # Unit buy increment
        sell_amount: float = 1.0 # Unit sell increment

    class Constraints(NamedTuple):
        R_max: float = 100 # Maximum battery capacity (100%)
    return Constraints, Decision, Policy, State


@app.cell
def _(Decision, State, jax):
    @jax.jit
    def contribution(state: State, decision: Decision) -> float:
        return state.p_t * (state.eta * decision.sell - decision.buy)
    return (contribution,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 8.3 - Modeling uncertainty

    How will we model the electricity prices? We will use an autoregressive model to model the price $p_{t+1}$ as a function of the previous prices

    $$p_{t+1} = \theta_{t_0}p_t + \theta_{t_1}p_{t-1} + \theta_{t_2}p_{t-2} + \epsilon_{t+1}$$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 8.4 - Designing policies

    We can solve the basic model using: (1) policy search, (2) lookahead policy, and (3) hybrid policy.

    ### 8.4.1 - Policy search

    #### Buy low, sell high

    A buy low, sell high policy works by buying energy to charge the battery when the price falls below a lower limit, and selling when the price goes above an upper limit.
    """
    )
    return


@app.cell
def _(Constraints, Decision, Policy, State, jax, jnp):
    @jax.jit
    def buy_low_sell_high(state: State, policy: Policy, constraints: Constraints) -> Decision:
        """A buy-low, sell-high policy.

        Charge the battery when the electricity price falls below a lower limit,
        and selling when the price goes above an upper limit.
        """
        can_buy = state.p_t <= policy.theta_buy
        can_sell = state.p_t >= policy.theta_sell
        can_hold = (state.p_t > policy.theta_buy) & (state.p_t < policy.theta_sell)

        max_buy = (constraints.R_max - state.R_t) / state.eta
        max_sell = state.R_t

        can_buy_amount = policy.buy_amount <= max_buy
        can_sell_amount = policy.sell_amount <= max_sell

        buy_amount = jnp.where(can_buy & can_buy_amount, policy.buy_amount, 0.0)
        sell_amount = jnp.where(can_sell & can_sell_amount, policy.sell_amount, 0.0)

        # Hold when in middle of price range OR can't execute the desired action
        hold_indicator = jnp.where(
            (can_hold | (can_buy & ~can_buy_amount) | (can_sell & ~can_sell_amount)),
            1.0,
            0.0
        )

        return Decision(buy=buy_amount, sell=sell_amount, hold=hold_indicator)
    return (buy_low_sell_high,)


@app.cell
def _(mo):
    mo.md(r"""## Demo""")
    return


@app.cell
def _(pl):
    prices = pl.read_excel(
        source="./data/energy_storage.xlsx",
        sheet_name="Raw Data"
    )

    rt_lmp = prices.select(["Day", "PJM RT LMP"])
    y_train = rt_lmp.select("PJM RT LMP").to_jax().flatten()[:180]
    y_test = rt_lmp.select("PJM RT LMP").to_jax().flatten()[-20:]
    return rt_lmp, y_test, y_train


@app.cell
def _(plt, rt_lmp):
    plt.figure(figsize=(16, 6))
    plt.plot(rt_lmp["Day"], rt_lmp["PJM RT LMP"])
    return


@app.cell
def _(ABC, abstractmethod, random):
    class UncertaintyModel(ABC):
        """Base class for modeling price uncertainty"""

        # @abstractmethod
        # def predict(self, key: jax.random.PRNGKey, state: State):
        #     """Predict (sample) the next price given the current state"""
        #     pass

        @abstractmethod
        def sample_posterior_predictive(self, key: random.PRNGKey, *model_args):
            """"""
            pass
    return (UncertaintyModel,)


@app.cell
def _(
    InferenceParams,
    MCMC,
    NUTS,
    Predictive,
    UncertaintyModel,
    dataclass,
    dist,
    jnp,
    numpyro,
    random,
    scan,
):
    @dataclass
    class AutoregressiveModel(UncertaintyModel):
        """AR(p) model"""

        lag_order: int = 2

        def __post_init__(self):
            self.posterior_samples = None
            self.fitted = False

        def model(self, y: jnp.ndarray, future: int = 0):
            """NumPyro model definition"""
            # TODO: This does not build the model using dynamic lag orders
            # Everything is hardcoded here to be an AR(2) process
            alpha_1 = numpyro.sample("alpha_1", dist.Normal(0, 1))
            alpha_2 = numpyro.sample("alpha_2", dist.Normal(0, 1))
            const = numpyro.sample("const", dist.Normal(0, 1))
            sigma = numpyro.sample("sigma", dist.HalfNormal(1))

            def transition(carry, _):
                y_prev, y_prev_prev = carry
                # AR process
                m_t = const + alpha_1 * y_prev + alpha_2 * y_prev_prev
                # `numpyro.handlers.condition` will use observed data where available and 
                # call numpyro.sample for future steps where observed data is not available
                y_t = numpyro.sample("y", dist.Normal(m_t, sigma))
                return (y_t, y_prev), m_t

            timesteps = jnp.arange(y.shape[0] - 2 + future)
            init = (y[1], y[0]) # Initial values are first two timesteps

            # Use handler to condition on observed data
            with numpyro.handlers.condition(data={"y": y[2:]}):
                _, mu = scan(transition, init, timesteps)

            # If forecasting, store forecasts in a deterministic site
            if future > 0:
                numpyro.deterministic("mu_forecast", mu[-future:])

            numpyro.deterministic("mu", mu)


        def fit(
            self, 
            key: random.PRNGKey,
            inference_params: InferenceParams,
            *model_args,
            **nuts_kwargs,
        ):
            rng_key, rng_subkey = random.split(key=key)

            nuts_kernel = NUTS(self.model, **nuts_kwargs)
            mcmc = MCMC(
                nuts_kernel,
                num_samples=inference_params.num_samples,
                num_warmup=inference_params.num_warmup,
                num_chains=inference_params.num_chains,
                progress_bar=False
            )

            mcmc.run(rng_subkey, *model_args)
            self.posterior_samples = mcmc.get_samples()

            self.fitted = True

        def sample_posterior_predictive(self, key: random.PRNGKey, *model_args):
            """Sample a series of price paths. These samples correspond to the 
            posterior predictive distribution.
            """
            rng_key, rng_subkey = random.split(key=key)
            predictive = Predictive(self.model, posterior_samples=self.posterior_samples)
            return predictive(rng_subkey, *model_args)
    return (AutoregressiveModel,)


@app.cell
def _(NamedTuple):
    class InferenceParams(NamedTuple):
        num_warmup: int = 250
        num_samples: int = 250
        num_chains: int = 10
    return (InferenceParams,)


@app.cell
def _(AutoregressiveModel, InferenceParams, random, y_train):
    ar_key = random.PRNGKey(seed=42)

    ar = AutoregressiveModel(lag_order=2)
    ip = InferenceParams(num_warmup=250, num_samples=250, num_chains=5)

    ar.fit(ar_key, ip, y_train)
    return ar, ar_key


@app.cell
def _(ar, ar_key, y_train):
    pps = ar.sample_posterior_predictive(ar_key, y_train)
    pps.keys()
    return (pps,)


@app.cell
def _(pps):
    y_mu = pps["mu"].mean(axis=0)
    y_std = pps["mu"].std(axis=0)
    return y_mu, y_std


@app.cell
def _(jnp, plt, y_mu, y_std, y_train):
    xs = jnp.arange(0, y_train.size)

    plt.figure(figsize=(16, 6))
    plt.plot(xs[2:], y_mu, label="In-sample")
    plt.plot(xs, y_train, label="Training")
    plt.fill_between(
        xs[2:], 
        y_mu + y_std * 3, 
        y_mu - y_std * 3,
        color="grey"
    )
    plt.legend()
    plt.show()
    return


@app.cell
def _(ar, ar_key, y_test, y_train):
    horizon = y_test.size

    forecasts = ar.sample_posterior_predictive(ar_key, y_train, horizon)

    test_mu = forecasts["mu_forecast"].mean(axis=0)
    test_std = forecasts["mu_forecast"].std(axis=0)
    return test_mu, test_std


@app.cell
def _(plt, test_mu, test_std, y_test):
    plt.figure(figsize=(16, 6))
    plt.plot(range(0, 20), test_mu, label="Forecast")
    plt.plot(range(0, 20), y_test, label="Test")
    plt.fill_between(
        range(0, 20),
        test_mu + test_std * 3,
        test_mu - test_std * 3,
        color="grey",
        alpha=0.25
    )
    plt.legend()
    plt.show()
    return


@app.cell
def _(Array, NamedTuple):
    class Result(NamedTuple):
        rewards: Array
        states: Array
        decisions: Array
    return (Result,)


@app.cell
def _(Constraints, Decision, State, jax, jnp):
    @jax.jit
    def transition(
        state: State,
        decision: Decision,
        constraints: Constraints,
        next_price: float
    ) -> State:
        # Energy update
        new_energy = state.R_t + (state.eta * decision.buy) - decision.sell
        constrained_energy = jnp.clip(new_energy, 0.0, constraints.R_max)

        # Price update
        new_price = next_price # From AR posterior predictive samples

        return State(R_t=constrained_energy, p_t=new_price, eta=state.eta)
    return (transition,)


@app.cell
def _(
    Array,
    Constraints,
    Policy,
    State,
    buy_low_sell_high,
    contribution,
    jax,
    jnp,
    transition,
):
    @jax.jit
    def simulate_policy(
        price_path: Array,
        policy: Policy,
        initial_state: State,
        constraints: Constraints
    ) -> float:

        def step(carry, price_info):
            state, total_reward = carry
            current_price, next_price = price_info

            # Update state
            current_state = state._replace(p_t=current_price)

            # Make decision
            decision = buy_low_sell_high(current_state, policy, constraints)

            # Compute contribution (objective)
            reward = contribution(current_state, decision)

            # Transition to next state
            next_state = transition(current_state, decision, constraints, next_price=next_price)

            return (next_state, total_reward + reward), (current_state, decision, reward)

        initial_total_reward = jnp.array([0.0])
        price_pairs = jnp.column_stack([price_path[:-1], price_path[1:]])

        (final_state, total_reward), trajectory = jax.lax.scan(
            step, 
            (initial_state, initial_total_reward), 
            price_pairs
        )

        return total_reward, trajectory[0], trajectory[1]
    return (simulate_policy,)


@app.cell
def _(Array, Constraints, Policy, Result, State, jax, simulate_policy, vmap):
    @jax.jit
    def evaluate_policy(
        sample_paths: Array,
        policy: Policy,
        state: State,
        constraints: Constraints,
    ) -> Result:
        simulate_vmap = vmap(simulate_policy, in_axes=(0, None, None, None))
        rewards, states, decisions = simulate_vmap(sample_paths, policy, state, constraints)

        return Result(
            rewards=rewards,
            states=states,
            decisions=decisions
        )
    return (evaluate_policy,)


@app.cell
def _(Array, Tuple, jnp):
    def create_parameter_grid(
        buy_min: float,
        buy_max: float,
        sell_min: float,
        sell_max: float,
        step: float
    ) -> Tuple[Array, Array]:
        buy_values = jnp.arange(buy_min, buy_max, step)
        sell_values = jnp.arange(sell_min, sell_max, step)
        buy_grid, sell_grid = jnp.meshgrid(buy_values, sell_values, indexing="ij")
        buy_grid, sell_grid = buy_grid.flatten(), sell_grid.flatten()

        # Sell price should not be below the buy price
        valid_mask = buy_grid < sell_grid
        valid_buy, valid_sell = buy_grid[valid_mask], sell_grid[valid_mask]

        return valid_buy, valid_sell
    return (create_parameter_grid,)


@app.cell
def _(Array, Constraints, Policy, State, evaluate_policy, jax, jnp, time):
    def grid_search(
        sample_paths: Array,
        buy_values: Array,
        sell_values: Array,
        state: State,
        constraints: Constraints,
    ):
        n_combinations = len(buy_values)

        def evaluate_pair(theta_buy, theta_sell):
            policy = Policy(theta_buy, theta_sell)
            res = evaluate_policy(sample_paths, policy, state, constraints)

            total_timesteps = res.decisions.buy.shape[1]
            buy_count = jnp.sum(res.decisions.buy, axis=1).flatten()
            sell_count = jnp.sum(res.decisions.sell, axis=1).flatten() 
            hold_count = jnp.sum(res.decisions.hold, axis=1).flatten()

            decision_stats = {
                "buy_frequency": buy_count / total_timesteps,
                "sell_frequency": sell_count / total_timesteps,
                "hold_frequency": hold_count / total_timesteps,
                "buy_count": buy_count,
                "sell_count": sell_count,
                "hold_count": hold_count
            }

            states_stats = {
                "mean_battery_capacity": res.states.R_t.mean(axis=0).flatten(),
                "std_battery_capacity": res.states.R_t.std(axis=0).flatten(),
            }

            return {
                "mean_cumsum_reward": res.rewards.mean().flatten()[0],
                "std_cumsum_reward": res.rewards.std().flatten()[0],
                **decision_stats,
                **states_stats
            }

        start = time.time()
        evaluate = jax.vmap(evaluate_pair, in_axes=(0, 0))
        res = evaluate(buy_values, sell_values)
        end = time.time()

        print(f"Simulation completed in: {(end - start):.2f} sec.")

        return res
    return (grid_search,)


@app.cell
def _(
    Constraints,
    State,
    create_parameter_grid,
    grid_search,
    jnp,
    pps,
    y_train,
):
    posterior_predictive_samples = pps["mu"]

    max_capacity = 100.
    init_capacity = 50. # 50% charge

    initial_battery_charge = jnp.array([init_capacity])
    initial_price = float(y_train[0])
    initial_state = State(
        R_t=initial_battery_charge, 
        p_t=initial_price, 
        eta=0.9
    )
    init_constraints = Constraints(R_max=max_capacity)

    buy_min = 10.
    buy_max = 100.
    sell_min = 10.
    sell_max = 100.
    step = 1

    buy_values, sell_values = create_parameter_grid(buy_min, buy_max, sell_min, sell_max, step)

    res = grid_search(
        posterior_predictive_samples, 
        buy_values, 
        sell_values, 
        initial_state, 
        init_constraints
    )
    return buy_values, res, sell_values


@app.cell
def _(res):
    res
    return


@app.function
def ticks_every_k(vals, k):
    vmin, vmax = min(vals), max(vals)
    return list(range((vmin // k) * k, vmax + 1, k))


@app.cell
def _(pl, plt, sns):
    def plot_contribution_heatmap(
        vectorized_results, 
        buy_values, 
        sell_values, 
        metric: str, 
        **kwargs
    ):
        n_combinations = len(buy_values)

        rows = []
        for i in range(n_combinations):
            theta_buy = int(buy_values[i])
            theta_sell = int(sell_values[i])
            metric_value = float(vectorized_results[metric][i])

            rows.append({
                "param1": theta_buy, 
                "param2": theta_sell, 
                metric: metric_value
            })

        df = pl.DataFrame(rows)

        uniq_param_1 = sorted(df.get_column("param1").unique().to_list())
        uniq_param_2 = sorted(df.get_column("param2").unique().to_list())

        # Pivot to wide form: rows=param1, cols=param2, values=metric
        heat_df = (
            df.pivot(index="param1", on="param2", values=metric)
              .sort("param1")
        )

        # Map numeric col names to strings, skip the index column "param1"
        col_map = {c: str(c) for c in heat_df.columns if c != "param1"}
        heat_df = heat_df.rename(col_map)

        # Build x_labels as strings to match renamed columns
        x_labels = [str(c) for c in uniq_param_2]
        heat_df = heat_df.select(["param1"] + x_labels)

        y_labels = heat_df.get_column("param1").to_list()
        matrix = heat_df.select(x_labels).to_numpy()

        # Flip vertically to make smaller param1 at bottom-left
        matrix = matrix[::-1, :]
        y_labels = y_labels[::-1]

        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(
            matrix, 
            annot=False, 
            xticklabels=x_labels, 
            yticklabels=y_labels
        )

        x_vals = [int(v) for v in x_labels]
        y_vals = [int(v) for v in y_labels]

        x_index_by_value = {v: i for i, v in enumerate(x_vals)}
        y_index_by_value = {v: i for i, v in enumerate(y_vals)}

        x_tick_vals = ticks_every_k(x_vals, 2)
        x_tick_pos = [x_index_by_value[v] for v in x_tick_vals if v in x_index_by_value]
        ax.set_xticks(x_tick_pos)
        ax.set_xticklabels([str(v) for v in x_tick_vals if v in x_index_by_value])

        y_tick_vals = ticks_every_k(y_vals, 2)
        y_tick_pos = [y_index_by_value[v] for v in y_tick_vals if v in y_index_by_value]
        ax.set_yticks(y_tick_pos)
        ax.set_yticklabels([str(v) for v in y_tick_vals if v in y_index_by_value])

        ax.set_xlabel("Sell")
        ax.set_ylabel("Buy")

        title = kwargs.get("title", None)
        plt.title(title)
        plt.show()
    return (plot_contribution_heatmap,)


@app.cell
def _(plt):
    def plot_battery_capacity(
        vectorized_results, 
        buy_values, 
        sell_values, 
        metric: str, 
        **kwargs
    ):
        n_combinations = len(buy_values)

        plt.figure(figsize=(16, 6))

        for i in range(0, n_combinations, max(1, n_combinations // 20)):  # Plot every Nth combination
            theta_buy = int(buy_values[i])
            theta_sell = int(sell_values[i])

            capacity_values = vectorized_results[metric][i]
            t = range(len(capacity_values))

            plt.plot(t, capacity_values, label=f"buy={theta_buy}, sell={theta_sell}", alpha=0.7)

        plt.xlabel("Time")
        plt.ylabel("Battery Capacity")
        plt.title("Battery Capacity Over Time (Sample Parameter Combinations)")

        if n_combinations <= 20:
            plt.legend()

        plt.show()
    return (plot_battery_capacity,)


@app.cell
def _(buy_values, plot_contribution_heatmap, res, sell_values):
    plot_contribution_heatmap(res, buy_values, sell_values, "mean_cumsum_reward")
    return


@app.cell
def _(buy_values, plot_contribution_heatmap, res, sell_values):
    plot_contribution_heatmap(res, buy_values, sell_values, "std_cumsum_reward")
    return


@app.cell
def _(buy_values, plot_battery_capacity, res, sell_values):
    plot_battery_capacity(res, buy_values, sell_values, "mean_battery_capacity")
    return


if __name__ == "__main__":
    app.run()
