"""Implementation of WaveNet Architecture."""

__maintainer__ = []

from aeon.networks.base import BaseDeepLearningNetwork


class WaveNet(BaseDeepLearningNetwork):
    """WaveNet architecture for autoregressive and probabilistic outputs.

    Implementation of the WaveNet architecture adapted for
    time series modeling with TensorFlow/Keras backend.
    """

    _config = {
        "python_dependencies": ["tensorflow"],
        "python_version": "<3.13",
        "structure": "encoder",
    }

    def __init__(
        self,
        kernel_size: int = 2,
        num_filter: int = 10,
        n_layers: int = 4,
    ):
        """Initialize the Wavenet architecture.

        Parameters
        ----------
        kernel_size : int, optional (default=2)
            Size of the convolutional kernel window for each dilated convolution.
        num_filter : int, optional (default=10)
            Number of convolutional filters in each dilated convolutional layer.
        n_layers : int, optional (default=4)
            Number of dilated convolutional layers to stack in each WaveNet block.

        """
        super().__init__()
        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.n_layers = n_layers

    def _wavenet_layer(
        self,
        input_tensor,
        num_filter: int,
        kernel_size: int,
        dilation_rate: int,
    ):
        """Single dilated conv layer in WaveNet.

        Parameters
        ----------
        input_tensor : tf.Tensor
            Input tensor of shape (batch_size, timesteps, channels).
        num_filter : int
            Number of filters for the dilated convolutions.
        kernel_size : int
            Size of the kernel used in dilated convolutions.
        dilation_rate : int
            Dilation rate for the dilated convolutions.


        Returns
        -------
        x_residual : tf.Tensor
            Output tensor after applying the residual connection.
        x_skip_connection : tf.Tensor
            Output tensor for the skip connection, to be aggregated at the block level.

        """
        import tensorflow as tf

        x_residual = input_tensor

        tanh_out = tf.keras.layers.Conv1D(
            filters=num_filter,
            kernel_size=kernel_size,
            padding="same",
            activation="tanh",
            dilation_rate=dilation_rate,
        )(input_tensor)
        sigm_out = tf.keras.layers.Conv1D(
            filters=num_filter,
            kernel_size=kernel_size,
            padding="same",
            activation="sigmoid",
            dilation_rate=dilation_rate,
        )(input_tensor)

        x = tf.keras.layers.Multiply()([tanh_out, sigm_out])

        x_skip_connection = tf.keras.layers.Conv1D(filters=1, kernel_size=1)(x)

        x_residual = tf.keras.layers.Add()([x_residual, x_skip_connection])

        return x_residual, x_skip_connection

    def _wavenet_block(
        self,
        input_tensor,
        num_filter: int,
        kernel_size: int,
        n_layers: int,
    ):
        """Wavenet_block with several wavenet layers.

        Parameters
        ----------
        input_tensor : tf.Tensor
            Input tensor for the block.
        num_filter : int
            Number of filters in each layer.
        kernel_size : int
            Size of the dilated convolution kernel.
        n_layers : int
            Number of WaveNet layers in the block (depth of dilation rates).

        Returns
        -------
        x : tf.Tensor
            Output tensor after applying dilated convolutions
            and residual connections.
        x_skip_connections : list of tf.Tensor
            List of skip connection tensors, to be summed
            and passed to the model output head.

        """
        import tensorflow as tf

        x = tf.keras.layers.Conv1D(filters=num_filter, kernel_size=1, padding="same")(
            input_tensor
        )

        x_skip_connections = []

        dilation_rates = [2**i for i in range(n_layers)]
        for dilation_rate in dilation_rates:
            x, x_skip_connection = self._wavenet_layer(
                input_tensor=x,
                num_filter=num_filter,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
            )
            x_skip_connections.append(x_skip_connection)

        return x, x_skip_connections

    def build_network(self, input_shape: tuple, **kwargs) -> tuple:
        """Build the complete WaveNet architecture.

        Parameters
        ----------
        input_shape : tuple of int
            Shape of the input data (timesteps, channels).
        **kwargs
            Additional keyword arguments (ignored in this implementation).

        Returns
        -------
        input_layer : tf.keras.layers.Input
            Keras Input tensor.
        output : tf.Tensor
            Output tensor after all WaveNet blocks and final dense output layer.

        """
        import tensorflow as tf

        input_layer = tf.keras.layers.Input(shape=input_shape)

        x = tf.keras.layers.Conv1D(
            filters=self.num_filter, kernel_size=1, padding="same"
        )(input_layer)

        x_residual, x_skip_connections = self._wavenet_block(
            input_tensor=x,
            num_filter=self.num_filter,
            kernel_size=self.kernel_size,
            n_layers=self.n_layers,
        )

        x_sum = tf.keras.layers.Add()(x_skip_connections)
        x = tf.keras.layers.Activation("relu")(x_sum)

        x = tf.keras.layers.Conv1D(filters=1, kernel_size=1, activation="relu")(x)
        x = tf.keras.layers.Conv1D(filters=1, kernel_size=1)(x)
        x = tf.keras.layers.Flatten()(x)
        output = tf.keras.layers.Dense(units=256, activation="softmax")(x)

        return input_layer, output
