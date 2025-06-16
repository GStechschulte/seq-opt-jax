import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# 2 - An Asset Selling Problem""")
    return


@app.cell
def _():
    from dataclasses import dataclass
    from functools import partial
    from typing import NamedTuple, Callable, Tuple

    import jax
    import jax.numpy as jnp
    from jax import random, jit, vmap, grad
    from jax.scipy.optimize import minimize as jax_minimize
    import marimo as mo
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    return Callable, NamedTuple, Tuple, jax, jnp, minimize, mo, random, vmap


@app.cell
def _(mo):
    mo.md(r"""## Part 1 - Base model""")
    return


@app.cell
def _(NamedTuple):
    class State(NamedTuple):
        holding: int # Whether we are holding the stock or not (0 = no, 1 = yes)
        price: float # If we sell, the price per share we recieve

    class Decision(NamedTuple):
        sell: int # 0 = hold, 1 = sell

    class Exogenous(NamedTuple):
        price_change: float # Change in price from previous period
        initial_price_mu: float = 50.0
        initial_price_std: float = 5.0

    class Policy(NamedTuple):
        policy_type: str
        theta_low: float = 30.0
        theta_high: float = 60.0
        theta_track: float = 5.0
        alpha: float = 0.1
    return Decision, Exogenous, Policy, State


@app.cell
def _(jax, jnp, random):
    def generate_price_change(
        key: jax.random.PRNGKey,
        shape: tuple,
        sigma: float = 1.0
    ) -> jnp.ndarray:
        """"""
        price = random.normal(key, shape) * sigma

        return price
    return (generate_price_change,)


@app.cell
def _(Decision, Exogenous, State):
    def transition(state: State, decision: Decision, exog: Exogenous) -> State:
        """"""
        # TODO: Constraint for x_t <= R_t_asset
        choice = state.holding - decision.sell
        next_price = state.price + exog.price_change

        # Update smoothed price
        # next_smoothed_price = (1 - alpha) * state.smoothed_price + alpha * next_price

        return State(
            holding=choice,
            price=next_price
        )
    return (transition,)


@app.cell
def _(Decision, State):
    def contribution(state: State, decision: Decision) -> float:
        """"""
        return state.price * decision.sell

    def is_valid_decision(state: State, decision: Decision) -> bool:
        """"""
        return decision.sell <= state.holding
    return (contribution,)


@app.cell
def _(Callable, Decision, Policy, State, jnp):
    def sell_low_policy(state: State, policy: Policy, t: int, T: int) -> Decision:
        """Sell if the price drops below the threshold or at a final time period."""
        can_sell = state.holding == 1
        should_sell = (state.price < policy.theta_low) | (t == T - 1)
        sell_decision = jnp.where(can_sell & should_sell, 1, 0)

        return Decision(sell=sell_decision)


    def high_low_policy(state: State, policy: Policy, t: int, T: int) -> Decision:
        """Sell if the price goes higher or lower than the high-low thresholds or at a final time period.
        """
        raise NotImplemented("This policy is not implemented.")


    def track_policy(state: State, policy: Policy, t: int, T: int) -> Decision:
        """Sell if the price rises above a tracking signal."""
        raise NotImplemented("This policy is not implemented.")


    def get_policy_function(policy_type: str) -> Callable:
        """Return the requested policy function."""
        policies = {
            "sell_low": sell_low_policy,
            "high_low": high_low_policy,
            "track": track_policy
        }

        return policies[policy_type]
    return (get_policy_function,)


@app.cell
def _(
    Exogenous,
    Policy,
    State,
    Tuple,
    contribution,
    get_policy_function,
    jax,
    jnp,
    transition,
):
    def simulate_single_path(
        initial_state: State,
        price_changes: jnp.ndarray,
        policy: Policy,
        T: int
    ) -> Tuple[float, jnp.ndarray, jnp.ndarray]:
        """Simulate a single sample path under a given policy."""
        policy_fn = get_policy_function(policy.policy_type)

        def step(carry, inputs):
            state, total_reward = carry
            price_change, t = inputs

            # Make a decision based on the current state
            decision = policy_fn(state, policy, t, T)

            # Compute contribution
            reward = contribution(state, decision)

            # Update state using transition functions
            exog = Exogenous(price_change=price_change)
            next_state = transition(state, decision, exog)

            return (next_state, total_reward + reward), (state, decision, reward)

        # Create time indices and combine with price changes
        t_idx = jnp.arange(T)
        inputs = (price_changes, t_idx)

        # Run simulation
        (final_state, total_reward), trajectory = jax.lax.scan(step, (initial_state, 0.0), inputs)

        return total_reward, trajectory[0], trajectory[1]

    return (simulate_single_path,)


@app.cell
def _(
    Policy,
    State,
    generate_price_change,
    jax,
    jnp,
    simulate_single_path,
    vmap,
):
    def evaluate_policy(
        key: jax.random.PRNGKey,
        N: int,
        T: int,
        initial_price: float,
        policy: Policy,
        sigma: float = 1.0
    ) -> dict:
        """"""
        initial_state = State(holding=1, price=initial_price)
        price_changes = generate_price_change(key, (N, T), sigma)

        simulate_vmap = vmap(simulate_single_path, in_axes=(None, 0, None, None))
        rewards, states, decisions = simulate_vmap(initial_state, price_changes, policy, T)

        return {
            "rewards": rewards,
            "mean": jnp.mean(rewards),
            "std": jnp.std(rewards),
            "states": states,
            "decisions": decisions
        }
    return (evaluate_policy,)


@app.cell
def _(Policy, Tuple, evaluate_policy, jax, jnp, minimize):
    def optimize_policy(
        key: jax.random.PRNGKey,
        policy_type: str,
        N: int = 1_000,
        T: int = 100,
        initial_price: float = 50.0,
        sigma: float = 1.0
    ) -> Tuple[Policy, float]:
        """"""
    
        def objective(params_arr):
            policy = Policy(policy_type=policy_type)
            res = evaluate_policy(key, N, T, initial_price, policy, sigma)
            return -res["mean"]

        x0 = jnp.array([45.0])
        bounds = [(30.0, 60.0)]

        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 50, "disp": True}
        )

        return res
    return


@app.cell
def _(Policy, State, evaluate_policy, generate_price_change, random):
    key = random.PRNGKey(42)

    N, T = 1_000, 100
    sigma = 1.0

    init_state = State(holding=1, price=50.0)
    init_policy = Policy(policy_type="sell_low", theta_low=45.0)

    price_changes = generate_price_change(key, (N, T), sigma)

    res = evaluate_policy(key, N, T, initial_price=50.0, policy=init_policy)

    # simulate_single_path(init_state, price_changes[0], init_policy, T)

    # optimize_policy(key, "sell_low")
    return (res,)


@app.cell
def _(res):
    res.keys()
    return


@app.cell
def _(res):
    res["rewards"][0]
    return


@app.cell
def _(res):
    res["decisions"].sell[0]
    return


@app.cell
def _(res):
    res["states"].holding[0]
    return


@app.cell
def _(res):
    res["states"].price[0]
    return


@app.cell
def _(mo):
    mo.md(r"""## Part 2 - Extensions""")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
