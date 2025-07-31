"""Tests for the WaveNet."""

import pytest

from aeon.networks import WaveNet
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "input_shape, filters, kernel_size, n_layers",
    [
        ((100, 1), 10, 2, 4),
        ((50, 3), 16, 3, 6),
        ((200, 5), 32, 2, 5),
        ((75, 2), 8, 4, 3),
        ((120, 1), 20, 3, 7),
    ],
)
def test_wavenet_build_network_different_params(
    input_shape, filters, kernel_size, n_layers
):
    """Test wavenet with different parameters."""
    import tensorflow as tf

    network = WaveNet(kernel_size=kernel_size, num_filter=filters, n_layers=n_layers)

    input_layer, output_layer = network.build_network(input_shape)

    assert input_layer.shape[1:] == input_shape
    assert len(output_layer.shape) == 2

    # creating model
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    assert model is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_wavenet_network_config():
    """Check configuration of WaveNet."""
    network = WaveNet()
    # Check _config attributes
    assert "python_dependencies" in network._config
    assert "tensorflow" in network._config["python_dependencies"]
    assert "python_version" in network._config
    assert "structure" in network._config
    assert network._config["structure"] == "encoder"
