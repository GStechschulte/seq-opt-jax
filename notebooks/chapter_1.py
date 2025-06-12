import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# 1 - Modeling Sequential Decision Problems""")
    return


@app.cell
async def _():
    from dataclasses import dataclass
    from functools import partial
    from typing import NamedTuple, Callable, Tuple

    import micropip
    await micropip.install("jax")
    await micropip.install("matplotlib")

    import jax
    import jax.numpy as jnp
    from jax import random, jit, vmap, grad
    from jax.scipy.optimize import minimize
    import marimo as mo
    import matplotlib.pyplot as plt

    return NamedTuple, Tuple, jax, jit, jnp, mo, partial, random, vmap


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 1.3.1 - A simple inventory problem

    A pizza restaurant has to decide how many pounds of sausage to order from its food distributor. The restaurant has to make the decision at the end of day t, communicate the order which then arrives the following morning to meet tomorrow’s orders. If there is sausage left over, it can be held to the following day. The cost of the sausage, and the price that it will be sold for the next day, is known in advance, but the demand is not.
    """
    )
    return


@app.cell
def _(NamedTuple):
    class InventoryParams(NamedTuple):
        """Immutable parameters for the inventory problem."""
        price: float          # p - selling price per pound
        cost: float           # c - purchase cost per pound
        mean_demand: float    # D_bar - mean daily demand
        std_demand: float     # sigma_D - standard deviation of demand
        initial_inventory: float = 0.0


    class PolicyParams(NamedTuple):
        """Parameters for the order-up-to policy."""
        theta_min: float      # Reorder point
        theta_max: float      # Order-up-to level


    class SimulationState(NamedTuple):
        """State during simulation."""
        inventory: float
        total_profit: float
        day: int
    return InventoryParams, PolicyParams, SimulationState


@app.cell
def _(InventoryParams, jax, jit, jnp, random):
    @jit
    def generate_demand_sample(key: jax.random.PRNGKey, params: InventoryParams) -> float:
        """Generate a single demand sample from a Normal distribution."""
        demand = random.normal(key, shape=()) * params.std_demand + params.mean_demand

        return jnp.maximum(demand, 0.0)
    return (generate_demand_sample,)


@app.cell
def _(
    InventoryParams,
    generate_demand_sample,
    jax,
    jit,
    jnp,
    partial,
    random,
    vmap,
):
    @partial(jit, static_argnums=(2,))
    def generate_demand_sequence(key: jax.random.PRNGKey, params: InventoryParams, num_days: int) -> jnp.ndarray:
        """Generate a sequence of demand samples."""
        keys = random.split(key, num_days)
        demands = vmap(lambda k: generate_demand_sample(k, params))(keys)
        return demands
    return (generate_demand_sequence,)


@app.cell
def _(PolicyParams, jit, jnp):
    @jit
    def order_up_to_policy(inventory: float, policy_params: PolicyParams) -> float:
        """Order-up-to policy from equation.

        Returns order quantity based on current inventory and policy parameters.
        """
        should_order = inventory < policy_params.theta_min
        order_quantity = jnp.where(
            should_order,
            policy_params.theta_max - inventory,
            0.0
        )

        return jnp.maximum(order_quantity, 0.0)
    return (order_up_to_policy,)


@app.cell
def _(jit, jnp):
    @jit
    def inventory_transition(inventory: float, order_qty: float, demand: float) -> float:
        """State transition function from equation.

        R_{t+1} = max{0, R_t + x_t - D_{t+1}}
        """
        return jnp.maximum(0.0, inventory + order_qty - demand)
    return (inventory_transition,)


@app.cell
def _(InventoryParams, jit, jnp):
    @jit
    def single_period_contribution(
        inventory: float,
        order_qty: float,
        demand: float,
        inv_params: InventoryParams
    ) -> float:
        """Single period contribution function."""
        available_inventory = inventory + order_qty
        sales = jnp.minimum(available_inventory, demand)
        contribution = -inv_params.cost * order_qty + inv_params.price * sales

        return contribution
    return (single_period_contribution,)


@app.cell
def _(
    InventoryParams,
    PolicyParams,
    SimulationState,
    inventory_transition,
    jit,
    order_up_to_policy,
    single_period_contribution,
):
    @jit
    def simulation_step(
        state: SimulationState,
        demand: float,
        policy_params: PolicyParams,
        inv_params: InventoryParams
    ) -> SimulationState:
        """Single step of the simulation."""
        # Make ordering decision
        order_qty = order_up_to_policy(state.inventory, policy_params)

        # Calculate contribution
        contribution = single_period_contribution(state.inventory, order_qty, demand, inv_params)

        # Update inventory
        new_inventory = inventory_transition(state.inventory, order_qty, demand)

        return SimulationState(
            inventory=new_inventory,
            total_profit=state.total_profit + contribution,
            day=state.day + 1
        )
    return (simulation_step,)


@app.cell
def _(
    InventoryParams,
    PolicyParams,
    SimulationState,
    Tuple,
    jax,
    jit,
    jnp,
    simulation_step,
):
    @jit
    def simulate_episode(
        demands: jnp.ndarray,
        policy_params: PolicyParams,
        inv_params: InventoryParams
    ) -> Tuple[float, jnp.ndarray]:
        """Simulate an entire episode given a demand sequence."""
        initial_state = SimulationState(
            inventory=inv_params.initial_inventory,
            total_profit=0.0,
            day=0
        )

        # Use scan for efficient sequential computation
        def step_fn(state, demand):
            new_state = simulation_step(state, demand, policy_params, inv_params)

            return new_state, new_state.inventory

        final_state, inventory_trajectory = jax.lax.scan(step_fn, initial_state, demands)

        return final_state.total_profit, inventory_trajectory
    return (simulate_episode,)


@app.cell
def _(
    InventoryParams,
    PolicyParams,
    generate_demand_sequence,
    jax,
    jit,
    jnp,
    partial,
    random,
    simulate_episode,
    vmap,
):
    @partial(jit, static_argnums=(3, 4))
    def evaluate_policy_batch(key: jax.random.PRNGKey, policy_params: PolicyParams,
                             inv_params: InventoryParams, num_scenarios: int,
                             num_days: int = 30) -> jnp.ndarray:
        """Evaluate policy across multiple scenarios using vectorization."""
        # Generate keys for each scenario
        keys = random.split(key, num_scenarios)

        # Generate demand sequences for all scenarios
        demand_sequences = vmap(
            lambda k: generate_demand_sequence(k, inv_params, num_days)
        )(keys)

        # Simulate all scenarios in parallel
        profits, _ = vmap(
            lambda demands: simulate_episode(demands, policy_params, inv_params)
        )(demand_sequences)

        return profits
    return (evaluate_policy_batch,)


@app.cell
def _(
    InventoryParams,
    PolicyParams,
    evaluate_policy_batch,
    jax,
    jit,
    jnp,
    partial,
):
    @partial(jit, static_argnums=(3,))
    def evaluate_policy_stats(
        key: jax.random.PRNGKey, 
        policy_params: PolicyParams,                     
        inv_params: InventoryParams, 
        num_scenarios: int,
        num_days: int = 30
    ) -> dict:
        """Evaluate policy and return statistics."""
        profits = evaluate_policy_batch(key, policy_params, inv_params, num_scenarios, num_days)

        return {
            'mean_profit': jnp.mean(profits),
            'std_profit': jnp.std(profits),
            'min_profit': jnp.min(profits),
            'max_profit': jnp.max(profits),
            'profits': profits
        }
    return


@app.cell
def _(
    InventoryParams,
    PolicyParams,
    Tuple,
    evaluate_policy_batch,
    jax,
    jit,
    jnp,
    partial,
    vmap,
):
    @partial(jit, static_argnums=(4, 5))
    def grid_search_policies(
        key: jax.random.PRNGKey,
        inv_params: InventoryParams,
        theta_min_range: jnp.ndarray,
        theta_max_range: jnp.ndarray,
        num_scenarios: int,
        num_days: int = 30
    ) -> Tuple[float, float, float]:
        """Vectorized grid search policy evaluation."""
        # Create pairwise combinations of thetas
        theta_min_grid, theta_max_grid = jnp.meshgrid(theta_min_range, theta_max_range)
        # Flatten for vectorization
        theta_min_flat = theta_min_grid.flatten()
        theta_max_flat = theta_max_grid.flatten()

        def evaluate_single_policy(theta_min, theta_max):
            policy_params = PolicyParams(theta_min=theta_min, theta_max=theta_max)
            profits = evaluate_policy_batch(key, policy_params, inv_params, num_scenarios, num_days)

            return jnp.where(theta_max > theta_min, jnp.mean(profits), -jnp.inf)

        profits = vmap(evaluate_single_policy)(theta_min_flat, theta_max_flat)

        # Find best policy
        best_idx = jnp.argmax(profits)
        best_theta_min = theta_min_flat[best_idx]
        best_theta_max = theta_max_flat[best_idx]
        best_profit = profits[best_idx]

        return best_theta_min, best_theta_max, best_profit
    return (grid_search_policies,)


@app.cell
def _(InventoryParams, grid_search_policies, jnp, random):
    # Initialize random key
    key = random.PRNGKey(42)
    key, subkey = random.split(key)

    inv_params = InventoryParams(
        price=8.0,
        cost=5.0,
        mean_demand=20.0,
        std_demand=5.0,
        initial_inventory=0.0
    )

    theta_min_range = jnp.linspace(0, 2 * inv_params.mean_demand, 8)
    theta_max_range = jnp.linspace(inv_params.mean_demand, 3 * inv_params.mean_demand, 8)

    best_theta_min, best_theta_max, best_profit = grid_search_policies(
        subkey, inv_params, theta_min_range, theta_max_range, 200, 30
    )
    return best_profit, best_theta_max, best_theta_min


@app.cell
def _(best_profit, best_theta_max, best_theta_min):
    best_theta_min, best_theta_max, best_profit
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 1.3.2 - A slighly more complicated inventory problem

    The simple inventory problem above has a simple state variable, i.e., only one variable (and is discrete). In this slightly more complicated inventory problem, more (and different) variables will be added to the state vector.

    **Step 1. Narrative**

    In this problem, the price paid for sausage is going to vary from day-to-day, where we assume the price on one day is independent of the price on the previous day. Then, we are going to be given a forecast of tomorrow's demand.

    **Step 2. Core elements**

    We would still like to maximize profits given by the sausage sales minus the cost of purchasing the sausage where the cost varies day-to-day. 

    **Step 3. Mathematical model**

    1.) State

    There are now three sources of uncertainty

    1. Difference between actual and forecasted demand.
    2. Demand forecast.
    3. Cost of sausage.

    We need to ensure that the state variable contains all the information needed by the objective function, transition function, and policy function.

    Initial state consists of:

    - Price
    - Initial inventory
    - Initial purchase cost
    - Initial forecast
    - Initial estimate demand uncertainty (std)
    - Initial estimate of the forecast uncertainty (std)

    We then have the information that changes over time, i.e., the dynamic state variable

    - Current inventory
    - Purchase cost
    - Demand forecast
    - Current estimate of demand uncertainty (std)
    - Current estimate of forecast uncertainty (std)

    2.) Decision variable

    How much we order at time $t$.

    3.) Exogenous information

    Now consists of:

    - Purchase costs
    - Forecasted demand
    - Actual demand (we assume the actual demand is a random deviation from the forecast)

    4.) Transition functions

    Specify how each of the five dynamic state variables in $S_t$ evolve over time.

    Update inventory according to

    $R_{t+1}^{inv} = \max(0, R_{t}^{inv} + x_t - \hat{D}_{t+1})$

    Actual demand is the forecasted demand plus the deviation $e_{t+1}^D$ from the forecast

    $\hat{D}_{t+1} = f_{t,t+1}^{D} + \epsilon_{t+1}^D$

    Forecasted demand is updated by

    $f_{t+1,t+2}^D = f_{t,t+1}^D + \epsilon_{t+1}^f$

    and the uncertainty of the actual and forecasted demand is updated by

    TODO...

    5.) Contribution

    The same as the previous example with the exception that the cost is now time dependent $c_t$.

    $C(S_t, x_t, \hat{D}_{t+1}) = -c_t * x_t + p \min(R_t + x_t, \hat{D}_{t+1})$

    **Step 4. Uncertainty model**

    Exogenous changes are described by Normal distributions.

    **Step 5. Designing policies**

    Instead of the order-up-to policy of the simpler model, we will order just enough to meet the expected demand for tomorrow, with an adjustment.

    $X^{\pi} (S_t|\theta) = \max(0, f_{t,t+1}^D - R_t) + \theta$

    If we had a perfect forecast, all we would have to do is order the difference between the forecasted demand and what is on inventory.

    **Step 6. Evaluating policies**

    This time we generate samples of all the random variables in the sequence $W_1, W_2,\ldots,W_T$ and $N$ samples of the entire sequence.
    """
    )
    return


@app.cell
def _(NamedTuple):
    class ComplexInventoryParams(NamedTuple):
        """Immutable parameters for the inventory problem."""
        price: float          # p - selling price per pound
        mean_cost: float           # c - purchase cost per pound
        std_cost: float
        mean_demand: float    # D_bar - mean daily demand
        std_demand: float     # sigma_D - standard deviation of demand
        std_forecast: float
        alpha: float = 0.1
        initial_inventory: float = 0.0
        initial_cost: float = 5.0
        initial_forecast: float = 20.0
        initial_sigma_D: float = 5.0
        initial_sigma_f: float = 2.0

    class ComplexState(NamedTuple):
        """Dynamic state variable S_t for the complex inventory problem"""
        inventory: float # R_inv_t
        cost: float # c_t
        forecast: float # f^D_{t,t+1}
        sigma_D: float # Current estimate of demand std
        sigma_f: float # Current estimate of forecast std


    class ExogenousInfo(NamedTuple):
        cost_next: float
        epsilon_f: float
        epsilon_D: float


    class ComplexPolicyParams(NamedTuple):
        """Parameters for the just-enough order policy."""
        theta: float
    return (
        ComplexInventoryParams,
        ComplexPolicyParams,
        ComplexState,
        ExogenousInfo,
    )


@app.cell
def _(ComplexInventoryParams, ExogenousInfo, jax, jit, jnp, random):
    @jit
    def generate_exogenous_samples(
        key: jax.random.PRNGKey, 
        params: ComplexInventoryParams,
        sigma_D: float,
        sigma_f: float
    ) -> ExogenousInfo:
        """Generates a single sample of exogenous information W_{t+1}
        """
        key1, key2, key3 = random.split(key, 3)

        # Purchase cost for the next period
        cost_next = random.normal(key1) * params.std_cost + params.mean_cost
        cost_next = jnp.maximum(cost_next, 0.0)
        # Change in forecast for next period
        epsilon_f = random.normal(key2) * sigma_f
        # Demand deviation from forecast
        epsilon_D = random.normal(key3) * sigma_D

        return ExogenousInfo(
            cost_next=cost_next,
            epsilon_f=epsilon_f,
            epsilon_D=epsilon_D
        )
    return (generate_exogenous_samples,)


@app.cell
def _(
    ComplexInventoryParams,
    ComplexState,
    Tuple,
    generate_exogenous_samples,
    jax,
    jnp,
    random,
):
    def generate_exogenous_sequence(
        key: jax.random.PRNGKey,
        params: ComplexInventoryParams,
        initial_state: ComplexState,
        num_days: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        """
        keys = random.split(key, 3)

        def step(carry, key_t):
            sigma_D, sigma_f = carry
            exog = generate_exogenous_samples(key_t, params, sigma_D, sigma_f)

            return (sigma_D, sigma_f), exog

        initial_carry = (initial_state.sigma_D, initial_state.sigma_f)
        _, exog_sequence = jax.lax.scan(step, initial_carry, keys)

        return exog_sequence.cost_next, exog_sequence.epsilon_f, exog_sequence.epsilon_D
    return (generate_exogenous_sequence,)


@app.cell
def _(ComplexPolicyParams, ComplexState, jit, jnp):
    @jit
    def complex_policy(
        state: ComplexState,
        policy_params: ComplexPolicyParams
    ) -> float:
        """
        """
        should_order = jnp.maximum(0.0, state.forecast - state.inventory)

        return should_order + policy_params.theta
    return (complex_policy,)


@app.cell
def _(ComplexInventoryParams, ComplexState, jit, jnp):
    @jit
    def complex_contribution(
        state: ComplexState,
        order_qty: float,
        actual_demand: float,
        params: ComplexInventoryParams
    ) -> float:
        """"""
        available_inventory = state.inventory + order_qty
        sales = jnp.minimum(available_inventory, actual_demand)
        contribution = -state.cost * order_qty + params.price * sales

        return contribution
    return (complex_contribution,)


@app.cell
def _(ComplexInventoryParams, ComplexState, ExogenousInfo, jit, jnp):
    @jit
    def complex_transition(
        state: ComplexState,
        exog: ExogenousInfo,
        inv_params: ComplexInventoryParams,
        order_qty: float
    ) -> ComplexState:
        """"""
        # Actual demand update
        actual_demand = state.forecast + exog.epsilon_D

        # Cost update
        new_cost = exog.cost_next

        # Forecasted demand update
        new_forecast = state.forecast + exog.epsilon_f

        # Inventory update
        new_inventory = jnp.maximum(0.0, state.inventory + order_qty - actual_demand)

        # Actual and forecasted demand uncertainty update
        demand_error = state.forecast - actual_demand
        new_sigma_D_sq = (
            (1 - inv_params.alpha) * 
            state.sigma_D ** 2 + 
            inv_params.alpha * 
            demand_error ** 2
        )
        new_sigma_D = jnp.sqrt(jnp.maximum(new_sigma_D_sq, 0.01))

        forecast_error = state.forecast - new_forecast
        new_sigma_f_sq = (
            (1 - inv_params.alpha) *
            state.sigma_f ** 2 +
            inv_params.alpha *
            forecast_error ** 2
        )
        new_sigma_f = jnp.sqrt(jnp.maximum(new_sigma_f_sq, 0.01))

        return ComplexState(
            inventory=new_inventory,
            cost=new_cost,
            forecast=new_forecast,
            sigma_D=new_sigma_D,
            sigma_f=new_sigma_f
        )
    return (complex_transition,)


@app.cell
def _(
    ComplexInventoryParams,
    ComplexPolicyParams,
    ComplexState,
    ExogenousInfo,
    Tuple,
    complex_contribution,
    complex_policy,
    complex_transition,
    jax,
    jit,
    jnp,
):
    @jit
    def simulate_complex_episode(
        costs: jnp.ndarray,
        epsilons_f: jnp.ndarray,
        epsilons_D: jnp.ndarray,
        policy_params: ComplexPolicyParams,
        inv_params: ComplexInventoryParams
    ) -> Tuple[float, jnp.ndarray]:
        """"""
        # Initialize state
        initial_state = ComplexState(
            inventory=inv_params.initial_inventory,
            cost=inv_params.initial_cost,
            forecast=inv_params.initial_forecast,
            sigma_D=inv_params.initial_sigma_D,
            sigma_f=inv_params.initial_sigma_f
        )

        def step(state, exog_tuple):
            cost_next, epsilon_f_next, epsilon_D_next = exog_tuple
            exog = ExogenousInfo(
                cost_next=cost_next,
                epsilon_f=epsilon_f_next,
                epsilon_D=epsilon_D_next
            )

            # Make decision using policy
            order_qty = complex_policy(state, policy_params)

            # Calculate actual demand for contribution
            actual_demand = state.forecast + epsilon_D_next
            actual_demand = jnp.maximum(actual_demand, 0.0)

            # Compute contribution
            contribution = complex_contribution(state, order_qty, actual_demand, inv_params)

            # Transition to next state
            new_state = complex_transition(state, exog, inv_params, order_qty)

            return new_state, (contribution, new_state.inventory, actual_demand)

        exog_sequence = (costs, epsilons_f, epsilons_D)

        final_state, (contributions, inventory_trajectory, demand_trajectory) = jax.lax.scan(
            step, initial_state, exog_sequence
        )

        total_profit = jnp.sum(contributions)

        return total_profit, inventory_trajectory
    return (simulate_complex_episode,)


@app.cell
def _(
    ComplexInventoryParams,
    ComplexPolicyParams,
    ComplexState,
    generate_exogenous_sequence,
    jax,
    jit,
    jnp,
    partial,
    random,
    simulate_complex_episode,
    vmap,
):
    @partial(jit, static_argnums=(3, 4))
    def evaluate_complex_policy_batch(
        key: jax.random.PRNGKey,
        policy_params: ComplexPolicyParams,
        inv_params: ComplexInventoryParams,
        num_scenarios: int,
        num_days: int = 30
    ) -> jnp.ndarray:
        # Initial state for generating exogenous sequences
        initial_state = ComplexState(
            inventory=inv_params.initial_inventory,
            cost=inv_params.initial_cost,
            forecast=inv_params.initial_forecast,
            sigma_D=inv_params.initial_sigma_D,
            sigma_f=inv_params.initial_sigma_f
        )

        keys = random.split(key, 3)

        def generate_scenario(key_scenario):
            return generate_exogenous_sequence(key_scenario, inv_params, initial_state, num_days)

        cost_seq, epsilon_f_seq, epsilon_D_seq = vmap(generate_scenario)(keys)

        def simulate_scenario(costs, epsilons_f, epsilons_D):
            profit, _ = simulate_complex_episode(
                costs, epsilons_f, epsilons_D, policy_params, inv_params
            )

            return profit

        profits = vmap(simulate_scenario)(cost_seq, epsilon_f_seq, epsilon_D_seq)

        return profits
    return (evaluate_complex_policy_batch,)


@app.cell
def _(
    ComplexInventoryParams,
    ComplexPolicyParams,
    Tuple,
    evaluate_complex_policy_batch,
    jax,
    jit,
    jnp,
    partial,
    vmap,
):
    @partial(jit, static_argnums=(3, 4))
    def grid_search_complex_policy(
        key: jax.random.PRNGKey,
        inv_params: ComplexInventoryParams,
        theta_range: jnp.ndarray,
        num_scenarios: int,
        num_days: int = 30
    ) -> Tuple[float, float]:
        """Vectorized grid search over policy parameters"""

        def evaluate_single_theta(theta):
            policy_params = ComplexPolicyParams(theta=theta)
            profits = evaluate_complex_policy_batch(
                key,
                policy_params,
                inv_params,
                num_scenarios,
                num_days
            )
            return jnp.mean(profits)

        profits = vmap(evaluate_single_theta)(theta_range)

        # Find best theta
        best_idx = jnp.argmax(profits)
        best_theta = theta_range[best_idx]
        best_profit = profits[best_idx]

        return best_theta, best_profit
    return (grid_search_complex_policy,)


@app.cell
def _(
    ComplexInventoryParams,
    ComplexPolicyParams,
    grid_search_complex_policy,
    jnp,
    random,
):
    complex_key = random.PRNGKey(42)

    complex_inv_params = ComplexInventoryParams(
        price=8.0,
        mean_cost=5.0,
        std_cost=0.5,
        mean_demand=20.0,
        std_demand=5.0,
        std_forecast=2.0,
        alpha=0.1,
        initial_inventory=0.0,
        initial_cost=5.0,
        initial_forecast=20.0,
        initial_sigma_D=5.0,
        initial_sigma_f=2.0
    )

    complex_policy_params = ComplexPolicyParams(theta=3.0)

    theta_range = jnp.linspace(0, 10, 100)

    theta, profit = grid_search_complex_policy(
        complex_key, 
        complex_inv_params,
        theta_range, 
        1000, 
        30
    )

    print(f"Best theta: {theta:.2f}, profit: {profit:.2f}")
    return


if __name__ == "__main__":
    app.run()
