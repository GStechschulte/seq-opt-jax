import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# 1 - Modeling Sequential Decision Problems""")
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
    import numpy as np
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


app._unparsable_cell(
    r"""
    @jit
    def generate_demand_sample(key: jax.random.PRNGKey, params: InventoryParams) -> float:
        \"\"\"Generate a single demand sample from a Normal distribution.\"\"
        demand = random.normal(key, shape=()) * params.std_demand + params.mean_demand

        return jnp.maximum(demand, 0.0)
    """,
    name="_"
)


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


if __name__ == "__main__":
    app.run()
