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
    from jax.scipy.optimize import minimize
    import marimo as mo
    import matplotlib.pyplot as plt
    return Callable, NamedTuple, Tuple, jax, jnp, mo, plt, random


@app.cell
def _(NamedTuple):
    class State(NamedTuple):
        holding: int # Whether we are holding the stock or not (0 = no, 1 = yes)
        price: float # If we sell, the price per share we recieve

    class Decision(NamedTuple):
        sell: int # 0 = hold, 1 = sell

    class Exogenous(NamedTuple):
        price_change: float # Change in price from previous period
        initial_price_mu: float = 40
        initial_price_std: float = 5.0

    class Policy(NamedTuple):
        policy_type: str
        theta_low: float = 0.0
        theta_high: float = 100.0
        theta_track: float = 0.0
        alpha: float = 0.1
    return Decision, Exogenous, Policy, State


@app.cell
def _(Exogenous, jax, random):
    def generate_price_change(
        key: jax.random.PRNGKey,
        shape: int,
        exog: Exogenous
    ) -> Exogenous:
        """"""
        price = random.normal(key, shape) * exog.initial_price_std + exog.initial_price_mu

        return Exogenous(price_change=price)
    return (generate_price_change,)


@app.function
def generate_exogenous_sample():
    pass


@app.cell
def _(Decision, Exogenous, State, jnp):
    def transition(state: State, decision: Decision, exog: Exogenous) -> State:
        """"""
        # TODO: Constraint for x_t <= R_t_asset
        choice = jnp.maximum(state.holding - decision.sell)
        next_price = state.price + exog.price_change

        # Update smoothed price
        # TODO

        return State(
            holding=choice,
            price=next_price
        )
    return (transition,)


@app.cell
def _(Decision, State):
    def contribution(state: State, decision: Decision) -> float:
        """"""
        return state.price * decision.sell * state.holding

    def is_valid_decision(state: State, decision: Decision) -> bool:
        """"""
        return decision.sell <= state.holding
    return (contribution,)


@app.cell
def _(Callable, Decision, Policy, State, jnp):
    def sell_low_policy(state: State, policy: Policy, t: int, T: int) -> Decision:
        """Sell if the price drops below the threshold or at a final time period."""
        sell_condition = (state.price < policy.theta_low) | (t == T - 1)
        sell_decision = jnp.where(sell_condition & (state.holding == 1), 1, 0)

        return Decision(sell=sell_decision)


    def high_low_policy(state: State, policy: Policy, t: int, T: int) -> Decision:
        """Sell if the price goes higher or lower than the high-low thresholds or at a final time period.
        """
        sell_condition = (
            ((state.price < policy.theta_low) | (state.price > policy.theta_high)) |
            (t == T - 1)
        )
        sell_decision = jnp.where(sell_condition & (state.holding == 1), 1, 0)

        return Decision(sell=sell_decision)


    def track_policy(state: State, policy: Policy, t: int, T: int) -> Decision:
        """Sell if the price rises above a tracking signal."""
        raise NotImplemented("This policy is not implemented yet.")


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
    
    return


@app.cell
def _(Exogenous, generate_price_change, plt, random):
    key = random.PRNGKey(42)

    shape = (10, 8)
    initial_price = 0.0

    exog = Exogenous(price_change=initial_price)

    exog = generate_price_change(key, shape, exog)



    plt.plot(exog.price_change)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
