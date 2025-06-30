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
    from enum import IntEnum
    from functools import partial
    from typing import NamedTuple, Callable, Tuple

    import jax
    import jax.numpy as jnp
    from jax import random, jit, vmap, grad
    from jax.scipy.optimize import minimize as jax_minimize
    import marimo as mo
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    return (
        Callable,
        NamedTuple,
        Tuple,
        dataclass,
        jax,
        jit,
        jnp,
        mo,
        random,
        vmap,
    )


@app.cell
def _(mo):
    mo.md(r"""## Part 1 - Base model""")
    return


@app.cell
def _(NamedTuple):
    class State(NamedTuple):
        holding: int # Whether we are holding the stock or not (0 = no, 1 = yes)
        price: float # If we sell, the price per share we recieve
        smoothed_price: float = 0.0

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
        """Price changes represent an incremental amount to be added to the
        current price to get a new price"""
        price_change = random.normal(key, shape) * sigma

        return price_change
    return (generate_price_change,)


@app.cell
def _(Decision, Exogenous, State):
    def transition(
        state: State, 
        decision: Decision, 
        exog: Exogenous,
        alpha: float = 0.1
    ) -> State:
        """"""
        # Update choice and prices
        choice = state.holding - decision.sell
        next_price = state.price + exog.price_change
        next_smoothed_price = (1 - alpha) * state.smoothed_price + alpha * next_price

        return State(
            holding=choice,
            price=next_price,
            smoothed_price=next_smoothed_price
        )
    return (transition,)


@app.cell
def _(Decision, State, jit):
    @jit
    def contribution(state: State, decision: Decision) -> float:
        """"""
        return state.price * decision.sell

    @jit
    def is_valid_decision(state: State, decision: Decision) -> bool:
        """"""
        return decision.sell <= state.holding
    return (contribution,)


@app.cell
def _(Callable, Decision, Policy, State, jnp, predictive_sell_low_policy):
    def sell_low_policy(state: State, policy: Policy, t: int, T: int) -> Decision:
        """Sell if the price drops below the threshold or at a final time period"""
        can_sell = state.holding == 1
        should_sell = (state.price < policy.theta_low) | (t == T - 1)
        sell_decision = jnp.where(can_sell & should_sell, 1, 0)

        return Decision(sell=sell_decision)


    def high_low_policy(state: State, policy: Policy, t: int, T: int) -> Decision:
        """Sell if the price goes higher or lower than the high-low thresholds or at a final time period"""
        can_sell = state.holding == 1
        should_sell = (
            (state.price < policy.theta_low) |
            (state.price > policy.theta_high) |
            (t == T - 1)
        )
        sell_decision = jnp.where(can_sell & should_sell, 1, 0)

        return Decision(sell=sell_decision)


    def track_policy(state: State, policy: Policy, t: int, T: int) -> Decision:
        """Sell if the price rises above a tracking signal"""
        can_sell = state.holding == 1
        should_sell = (
            (state.price >= state.smoothed_price + policy.theta_track) |
            (t == T - 1)
        )
        sell_decision = jnp.where(can_sell & should_sell, 1, 0)

        return Decision(sell=sell_decision)


    def get_policy_function(policy_type: str) -> Callable:
        """Return the requested policy function"""
        policies = {
            "sell_low": sell_low_policy,
            "high_low": high_low_policy,
            "track": track_policy,
            "predictive": predictive_sell_low_policy
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
        """Simulate a single sample path under a given policy"""
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
def _(Policy, Tuple, evaluate_policy, jax):
    def optimize_policy(
        key: jax.random.PRNGKey,
        policy_type: str,
        N: int = 1_000,
        T: int = 100,
        initial_price: float = 50.0,
        sigma: float = 1.0
    ) -> Tuple[Policy, float]:
        """"""
        from scipy.optimize import minimize

        def objective(params):
            if policy_type == "sell_low":
                policy = Policy(policy_type=policy_type, theta_low=params[0])
            elif policy_type == "high_low":
                policy = Policy(policy_type=policy_type, theta_low=params[0], theta_high=params[1])
            elif policy_type == "track":
                policy = Policy(policy_type=policy_type, theta_track=params[0])

            res = evaluate_policy(key, N, T, initial_price, policy, sigma)

            return -res["mean"]  # Negative because we want to maximize

        # Set initial parameters and bounds based on policy type
        if policy_type == "sell_low":
            x0 = [45.0]
            bounds = [(30.0, 60.0)]
        elif policy_type == "high_low":
            x0 = [40.0, 80.0]
            bounds = [(30.0, 55.0), (55.0, 70.0)]
        elif policy_type == "track":
            x0 = [5.0]
            bounds = [(1.0, 10.0)]

        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 50}
        )

        # Create optimized policy
        if policy_type == "sell_low":
            opt_policy = Policy(policy_type=policy_type, theta_low=res.x[0])
        elif policy_type == "high_low":
            opt_policy = Policy(policy_type=policy_type, theta_low=res.x[0], theta_high=res.x[1])
        elif policy_type == "track":
            opt_policy = Policy(policy_type=policy_type, theta_track=res.x[0])

        return opt_policy, -res.fun
    return (optimize_policy,)


@app.cell
def _(Policy, State, evaluate_policy, generate_price_change, random):
    key = random.PRNGKey(42)

    N, T = 1_000, 100
    sigma = 1.0

    init_state = State(holding=1, price=50.0)
    init_policy = Policy(policy_type="sell_low", theta_low=50.0)

    price_changes = generate_price_change(key, (N, T), sigma)

    res = evaluate_policy(key, N, T, initial_price=50.0, policy=init_policy)
    return N, T, key, price_changes


@app.cell
def _(N, Policy, T, evaluate_policy, key, optimize_policy):
    # Test basic policy evaluation
    policy = Policy(policy_type="sell_low", theta_low=50.0)
    results = evaluate_policy(key, N, T, 50.0, policy)
    print("Non-optimized")
    print("=" * 15)
    print(f"Sell-low policy mean return: {results['mean']:.2f}, theta low = {policy.theta_low}")

    # Test policy optimization
    opt_policy, opt_return = optimize_policy(key, "sell_low")
    print("\nOptimized")
    print("=" * 15)
    print(f"Sell-low policy mean return: {opt_return:.2f}, theta low = {opt_policy.theta_low}")
    return policy, results


@app.cell
def _(N, Policy, T, evaluate_policy, key, optimize_policy, policy, results):
    # Test basic policy evaluation
    hl_policy = Policy(policy_type="high_low", theta_low=40.0, theta_high=80.0)
    hl_results = evaluate_policy(key, N, T, 50.0, hl_policy)
    print("Non-optimized")
    print("=" * 15)
    print(f"high_low policy mean return: {results['mean']:.2f}, theta low = {policy.theta_low}")

    # Test policy optimization
    hl_opt_policy, hl_opt_return = optimize_policy(key, "high_low")
    print("\nOptimized")
    print("=" * 15)
    print(f"high_low policy mean return: {hl_opt_return:.2f}, theta low = {hl_opt_policy.theta_low}, theta high = {hl_opt_policy.theta_high}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Part 2 - Extensions

    ### Time series processes

    Often, we will want a more realistic price process that captures autocorrelation over time. We may use a machine learning model to learn the autoregressive component of the time series process. Assume the time series process is given by

    $$p_{t+1} = \bar{\eta}_{t_0}p_t + \bar{\eta}_{t_1}p_{t-1} + \epsilon_{t+1}$$

    where $\epsilon \sim N(0, 4^2)$ and where $\bar{\eta}_t = (\bar{\eta}_{t_0},\bar{\eta}_{t_1})$ is the estimate of $\eta$ given what we know at time $t$.

    The incorporation of this price model requires us to change the following

    - Transition function.
    - Policy.
    """
    )
    return


@app.cell
def _():
    import numpyro
    import numpyro.distributions as dist

    from numpyro.infer import MCMC, NUTS, Predictive
    from numpyro.contrib.control_flow import scan
    return MCMC, NUTS, Predictive, dist, numpyro, scan


@app.cell
def _(NamedTuple, dataclass, jnp):
    class ARState(NamedTuple):
        """Extended state for AR model"""
        holding: int
        price: float
        price_history: jnp.ndarray  # Last k prices for AR model
        predicted_next_price: float
        prediction_uncertainty: float
        smoothed_price: float = 0.0

    class ARExogenous(NamedTuple):
        """Exogenous info for AR model"""
        noise: float
        ar_params: jnp.ndarray
        noise_std: float = 1.0

    @dataclass
    class ARModelConfig:
        """Configuration for AR model"""
        lag_order: int = 2
        noise_std_prior: float = 2.0
        ar_param_prior_std: float = 0.5

    class ARPolicy(NamedTuple):
        """Policy that can use AR predictions"""
        policy_type: str
        theta_low: float = 30.0
        theta_high: float = 60.0
        theta_track: float = 5.0
        use_prediction: bool = True  # Whether to use AR predictions
        prediction_weight: float = 1.0  # How much to weight predictions
        uncertainty_penalty: float = 0.1  # Penalty for uncertainty
        alpha: float = 0.1
    return ARModelConfig, ARPolicy, ARState


@app.cell
def _(jax, jnp, price_changes, random):
    def generate_ar_data(
        key: jax.random.PRNGKey,
        T: int = 200,
        true_params: jnp.ndarray = jnp.array([0.7, 0.2]),
        noise_std: float = 1.0,
        initial_price: float = 50.0
    ) -> jnp.ndarray:
        """Simulate data according to an AR(2) process"""
        prices = jnp.zeros(T)
        prices = prices.at[0].set(initial_price)
        prices = prices.at[1].set(initial_price + random.normal(key, ()) * noise_std)

        noise_key, _ = random.split(key)
        noise = random.normal(noise_key, (T - 2, )) * noise_std

        for t in range(2, T):
            ar_component = jnp.sum(true_params * price_changes[t - 2: t][::-1])
            prices = prices.at[t].set(ar_component + noise[t - 2])

        return prices
    return


@app.cell
def _(dist, jnp, numpyro, scan):
    def ar_price_model(
        prices: jnp.ndarray,
        lag_order: int = 2
    ):
        """NumPyro AR model for price evolution"""
        T = len(prices)

        # Define priors
        const = numpyro.sample("const", dist.Normal(0, 1))
        ar_params = numpyro.sample("ar_params", dist.Normal(0, 0.5).expand([lag_order]))
        noise_std = numpyro.sample("noise_std", dist.HalfNormal(2.0))

        def transition(carry, _):
            # carry holds the N (lag-order) previous prices needed for the next prediction
            lagged_prices = carry

            # Calculate the next price
            next_price_mu = const + jnp.dot(ar_params, lagged_prices)

            # Sample the next obs
            next_price_obs = numpyro.sample("price", dist.Normal(next_price_mu, noise_std))

            # Update carry
            new_carry = jnp.concatenate([jnp.array([next_price_obs]), lagged_prices[:-1]])

            return new_carry, next_price_mu

        # Initial carry contains the first N (lag-order) prices, reversed to align with
        # the AR params
        init_carry = prices[lag_order - 1::-1]

        # The data to be observed by the model starts after the initial lag period.
        observed_prices = prices[lag_order:]

        with numpyro.handlers.condition(data={"price": observed_prices}):
            timesteps = jnp.arange(T - lag_order)
            final_carry, means = scan(transition, init_carry, timesteps)

        numpyro.deterministic("predicted_means", means)

    return (ar_price_model,)


@app.cell
def _(ARModelConfig, MCMC, NUTS, Tuple, ar_price_model, jax, jnp):
    class ARModel:
        """An autoregressive model that makes real-time predictions"""
        def __init__(self, config: ARModelConfig):
            self.config = config
            self.posterior_samples = None

        def fit(
            self, 
            key: jax.random.PRNGKey, 
            prices: jnp.ndarray, 
            num_chains=10,
            num_samples: int = 500, 
            num_warmup: int = 250
        ):
            """Fit the AR model using NUTS"""
            kernel = NUTS(ar_price_model)
            mcmc = MCMC(
                kernel, 
                num_chains=num_chains, 
                num_warmup=num_warmup, 
                num_samples=num_samples
            )
            mcmc.run(key, prices, self.config.lag_order)
            self.posterior_samples = mcmc.get_samples()

            return self.posterior_samples

        def predict(
            self,
            key: jax.random.PRNGKey,
            price_history: jnp.ndarray,
            num_samples: int = 500
        ) -> Tuple[float, float]:
            """Predict the next price given the current price history"""

            raise NotImplementedError("This method is not implemented...")
    return


@app.cell
def _(
    ARState,
    Decision,
    Exogenous,
    PredictiveState,
    jax,
    jnp,
    predicted_price,
    prediction_std,
    random,
):
    def predictive_transition(
        key: jax.random.PRNGKey,
        state: ARState,
        decision: Decision,
        exog: Exogenous,
        model,
        alpha: float = 0.1
    ) -> ARState:
        """Transition function that includes the ARModel prediction for the price process"""
        # Update holding
        next_holding = state.holding - decision.sell
        # Current price evolves according to exogenous process
        next_price = state.price + exog.price_change
        # Update price history
        new_history = jnp.concatenate([
            state.price_history[1:],
            jnp.array([next_price])
        ])
        # Make prediction for the price after next_price (t+2 prediction when at t+1)
        pred_key, _ = random.split(key)
        predicted_mu, predicted_std = model.predict(pred_key, new_history)

        # Update smoothed price
        next_smoothed_price = (1 - alpha) * state.smoothed_price + alpha * next_price

        return PredictiveState(
            holding=next_holding,
            price=next_price,
            price_history=new_history,
            predicted_next_price=predicted_price,
            prediction_uncertainty=prediction_std,
            smoothed_price=next_smoothed_price
        )
    return (predictive_transition,)


@app.cell
def _(ARPolicy, ARState, pred):
    def predictive_sell_low_policy(
        state: ARState,
        policy: ARPolicy,
        t: int,
        T: int
    ):
        """Sell-low policy that considers next price prediction"""
        can_sell = state.holding == 1

        if policy.use_prediction and t < T - 1:
            # Consider both current price and prediction
            current_signal = state.price < policy.theta_low

            # If prediction suggests that price will go even lower, sell now
            # Use uncertainty to adjust thresholds
            adjusted_threshold = (
                policy.theta_low - policy.uncertainty_penalty * state.prediction_uncertainty
            )
            prediction_signal = state.predicted_next_price < adjusted_threshold

            # Combine signals with weighting
            should_sell = current_signal | (policy.prediction_weight * pred)
        else:
            should_sell = state.price < policy.theta_low
    return (predictive_sell_low_policy,)


@app.cell
def _(
    ARPolicy,
    ARState,
    Exogenous,
    Tuple,
    contribution,
    get_policy_function,
    jax,
    jnp,
    predictive_transition,
    random,
):
    def simulate_with_predictions(
            initial_state: ARState,
            price_changes: jnp.ndarray,
            policy: ARPolicy,
            model,
            key: jax.random.PRNGKey,
            T: int
        ) -> Tuple[float, jnp.ndarray]:
            """Simulate with real-time AR predictions at each step."""
            policy_fn = get_policy_function(policy.policy_type)

            def step(carry, inputs):
                state, total_reward, step_key = carry
                price_change, t = inputs

                # Split key for this step
                decision_key, pred_key, next_key = random.split(step_key, 3)

                # Make decision using current state (which includes prediction)
                decision = policy_fn(state, policy, t, T)

                # Compute reward
                reward = contribution(state, decision)

                # Transition to next state (includes making new prediction)
                exog = Exogenous(price_change=price_change)
                next_state = predictive_transition(
                    state, decision, exog, model, pred_key, policy.alpha
                )

                return (next_state, total_reward + reward, next_key), (state, decision, reward)

            # Run simulation
            t_indices = jnp.arange(T)
            inputs = (price_changes, t_indices)
            keys = random.split(key, T + 1)

            (final_state, total_reward, _), trajectory = jax.lax.scan(
                step, 
                (initial_state, 0.0, keys[0]), 
                inputs,
                keys[1:]
            )

            return total_reward, trajectory
    return (simulate_with_predictions,)


@app.cell
def _(
    ARPolicy,
    ARState,
    generate_price_change,
    jax,
    jnp,
    random,
    simulate_with_predictions,
):
    # NEW: Evaluation function with real-time AR predictions
    def evaluate_predictive_policy(
        key: jax.random.PRNGKey,
        model,
        N: int,
        T: int,
        initial_prices: jnp.ndarray,
        policy: ARPolicy,
        sigma: float = 1.0
    ) -> dict:
        """
        Evaluate policy using real-time AR predictions.
        """
        # Generate simple price changes for the "true" process
        # (These represent the actual market movements, not AR predictions)
        sim_key, eval_key = random.split(key)
        price_changes = generate_price_change(sim_key, (N, T), sigma)

        # Create initial state with prediction
        pred_key, sim_keys = random.split(eval_key)
        initial_pred, initial_std = model.predict(
            initial_prices, pred_key
        )

        initial_state =  ARState(
            holding=1,
            price=initial_prices[-1],
            price_history=initial_prices,
            predicted_next_price=initial_pred,
            prediction_uncertainty=initial_std,
            smoothed_price=initial_prices[-1]
        )

        # Simulate with predictions
        sim_keys_split = random.split(sim_keys, N)

        def simulate_one_path(path_key, price_changes_path):
            return simulate_with_predictions(
                initial_state, price_changes_path, policy, 
                model, path_key, T
            )[0]  # Just return reward

        rewards = jax.vmap(simulate_one_path)(sim_keys_split, price_changes)

        return {
            "rewards": rewards,
            "mean": jnp.mean(rewards),
            "std": jnp.std(rewards),
        }
    return


@app.cell
def _(jnp, random):
    rng = random.PRNGKey(456)
    sim_length = 100
    lag_order = 2
    y = jnp.zeros(sim_length + lag_order)
    y = y.at[0].set(50.)
    y = y.at[1].set(51.)
    return lag_order, rng, y


@app.cell
def _(Predictive, ar_price_model, lag_order, random, rng, y):
    prior_predictive = Predictive(model=ar_price_model, num_samples=1)
    rng_key, rng_subkey = random.split(rng)
    prior_samples = prior_predictive(rng_subkey, prices=y, lag_order=lag_order)


    return (prior_samples,)


@app.cell
def _(prior_samples):
    prior_samples
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
