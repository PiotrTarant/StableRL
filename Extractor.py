import torch as th
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.policies import ActorCriticPolicy


class HybridCNNMLPFeatureExtractor(BaseFeaturesExtractor):
    """
    Niestandardowy ekstraktor funkcji, który łączy CNN i MLP.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super(HybridCNNMLPFeatureExtractor, self).__init__(observation_space, features_dim)

        self.device = get_device("auto")

        # Obsługa przestrzeni obserwacji typu Dict
        if isinstance(observation_space, gym.spaces.Dict):
            # Zakładamy, że dane obrazu znajdują się pod kluczem "image"
            observation_space = observation_space["stable"]

        # Zdefiniowanie części CNN (dla danych 2D)
        self.cnn = th.nn.Sequential(
            th.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            th.nn.ReLU(),
            th.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # Dodano padding
            th.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            th.nn.ReLU(),
            th.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # Kolejny padding
        )

        # Dopasowanie wejścia do CNN
        with th.no_grad():
            # Przykładowa próbka z observation_space
            example_input = observation_space.sample()

            # Jeśli próbka to dict, wyciągnij odpowiednie dane (np. "image")
            if isinstance(example_input, dict):
                example_input = example_input["stable"]

            # Obsługa wymiarów (upewnienie się, że dane są 4D: [batch, channels, height, width])
            if len(example_input.shape) == 2:  # Wejście 2D: (H, W)
                example_input = example_input[None, None, :, :]  # Dodanie [batch, channels]
            elif len(example_input.shape) == 3:  # Wejście 3D (np. RGB obrazy)
                example_input = example_input[None, :, :, :]  # Dodanie [batch]
            else:
                raise ValueError(f"Unexpected observation shape: {example_input.shape}. Expected 2D or 3D data.")

            # Przekształcenie na tensor
            example_input = th.as_tensor(example_input, dtype=th.float32).to(self.device)

            # Przepływ danych przez CNN i obliczenie wymiaru wyjściowego
            cnn_output_dim = self.cnn(example_input).view(-1).shape[0]

        # Zdefiniowanie części MLP
        self.mlp = th.nn.Sequential(
            th.nn.Linear(cnn_output_dim, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, features_dim),
            th.nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Jeśli obserwacje to dict, wyodrębnij dane z klucza "image"
        if isinstance(observations, dict):
            observations = observations["stable"]

        # Poprawne dopasowanie wymiarów wejściowych
        if len(observations.shape) == 3:  # [batch, H, W], dodaj kanał
            observations = observations[:, None, :, :]
        elif len(observations.shape) != 4:  # Oczekujemy [batch, channels, H, W]
            raise ValueError(f"Unexpected input shape: {observations.shape}. Expected 4D data.")

        # Przepływ danych przez CNN -> MLP
        cnn_out = self.cnn(observations)
        cnn_out_flattened = th.flatten(cnn_out, start_dim=1)
        return self.mlp(cnn_out_flattened)


# Niestandardowa polityka PPO wykorzystująca hybrydowy ekstraktor funkcji
class HybridCNNMLPPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=HybridCNNMLPFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=128),
        )
