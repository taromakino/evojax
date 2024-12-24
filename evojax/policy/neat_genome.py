import jax
import jax.numpy as jnp
from evojax.algo.neat import INT_TO_ACTIVATION, INT_TO_AGGREGATION
from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.task.base import TaskState
from typing import Tuple


class NEATPolicy(PolicyNetwork):
    def __init__(self, num_inputs, num_outputs, max_nodes, max_connections_per_node):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.max_nodes = max_nodes
        self.max_connections_per_node = max_connections_per_node
        self._forward_fn = jax.vmap(self.forward, in_axes=(0, 0))

    def forward(self,
                params: jnp.ndarray,
                x: jnp.ndarray) -> jnp.ndarray:
        out = jnp.zeros(self.max_nodes)
        out = out.at[-self.num_inputs:].set(x)
        for i in range(self.max_nodes):
            node_aggr = 0.
            bias, response = params[i, 0], params[i, 1]
            for j in range(2, self.max_connections_per_node - 1, 2):
                node_in, weight = params[i, j], params[i, j + 1]
                node_aggr += jnp.where(jnp.isnan(node_in), 0., out[node_in.astype(int)] * weight)
            out = out.at[i].set(jnp.tanh(bias + response * node_aggr))
        return out[:self.num_outputs]

    def get_actions(self,
                    t_states: TaskState,
                    params: jnp.ndarray,
                    p_states: PolicyState) -> Tuple[jnp.ndarray, PolicyState]:
        return self._forward_fn(params, t_states.obs), p_states