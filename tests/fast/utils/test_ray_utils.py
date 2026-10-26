from unittest.mock import patch

import pytest
from ray._common.constants import HEAD_NODE_RESOURCE_NAME

from miles.utils.ray_utils import compute_ray_pin_head_options


def test_pin_head_uses_live_gcs_node_without_dashboard():
    head_id = "a" * 56
    nodes = [
        {"Alive": False, "NodeID": "old-head", "Resources": {HEAD_NODE_RESOURCE_NAME: 1}},
        {"Alive": True, "NodeID": "worker", "Resources": {"GPU": 8}},
        {"Alive": True, "NodeID": head_id, "Resources": {HEAD_NODE_RESOURCE_NAME: 1}},
    ]
    with patch("miles.utils.ray_utils.ray.nodes", return_value=nodes), patch(
        "ray.util.state.list_nodes", side_effect=AssertionError("Dashboard API must not be used")
    ):
        strategy = compute_ray_pin_head_options()["scheduling_strategy"]
    assert strategy.node_id == head_id
    assert strategy.soft is False


@pytest.mark.parametrize(
    "nodes",
    [[], [{"Alive": False, "NodeID": "old-head", "Resources": {HEAD_NODE_RESOURCE_NAME: 1}}]],
)
def test_pin_head_rejects_missing_live_head(nodes):
    with patch("miles.utils.ray_utils.ray.nodes", return_value=nodes):
        with pytest.raises(RuntimeError, match="Could not find a head node"):
            compute_ray_pin_head_options()
