import torch as th
import gymnasium as gym
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_device

class CNNMLPFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, original_observation_space: gym.spaces.Dict,
                 features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        self.device = get_device("auto")

        # Extracting shapes dynamically
        self.stable_shape = original_observation_space["stable"].shape
        self.grid_contents_shape = original_observation_space["grid_contents"].shape
        self.horse_list_shape = original_observation_space["horse_list"].shape
        self.agent_position_shape = original_observation_space["agent_position"].shape
        self.current_horse_index_shape = original_observation_space["current_horse_index"].shape

        self.stable_size = np.prod(self.stable_shape)
        self.grid_contents_size = np.prod(self.grid_contents_shape)
        self.horse_list_size = np.prod(self.horse_list_shape)
        self.agent_position_size = np.prod(self.agent_position_shape)
        self.current_horse_index_size = np.prod(self.current_horse_index_shape)

        self.input_dim = self.stable_size + self.grid_contents_size + self.horse_list_size + self.agent_position_size + self.current_horse_index_size

        assert self.input_dim == observation_space.shape[0], "Observation space size mismatch!"

        num_in_channels = 1 + self.grid_contents_shape[-1] + 1  # stable + grid_contents + agent position
        self.cnn_combined = th.nn.Sequential(
            th.nn.Conv2d(num_in_channels, 32, kernel_size=3, padding=1),
            th.nn.ReLU(),
            th.nn.AdaptiveAvgPool2d(output_size=(4, 4)),
            th.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            th.nn.ReLU(),
            th.nn.AdaptiveAvgPool2d(output_size=(2, 2))
        )
        # CNN for horse_list
        self.cnn_horse_list = th.nn.Sequential(
            th.nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
            th.nn.ReLU(),
            th.nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1),
            th.nn.ReLU(),
            th.nn.Flatten()
        )

        print(f"Self.grid_contents_shape: {self.grid_contents_shape}")
        print(
            f"Shape tensor generated for cnn_grid_contents: {(1, self.grid_contents_shape[-1], *self.grid_contents_shape[:-1])}")

        # Compute output dimensions dynamically
        cnn_output_dim_combined = self.cnn_combined(
            th.zeros(1, num_in_channels, *self.stable_shape)
        ).view(-1).shape[0]
        cnn_horse_list_out_size = self.cnn_horse_list(
            th.zeros(1, 8, self.horse_list_shape[0])
        ).view(-1).shape[0]
        # MLP for other data
        other_input_dim = int(self.current_horse_index_size)
        self.mlp_other = th.nn.Sequential(
            th.nn.Linear(other_input_dim, 64),
            th.nn.ReLU(),
            th.nn.Linear(64, 32),
            th.nn.ReLU()
        )

        # Final MLP combining all features
        self.mlp = th.nn.Sequential(
            th.nn.Linear(cnn_output_dim_combined + cnn_horse_list_out_size + 32, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, features_dim),
            th.nn.ReLU()
        )



    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Extract stable (reshaped as 2D grid with a single channel)
        stable = observations[:, :self.stable_size].view(-1, 1, *self.stable_shape)


        # Extract and correctly reshape grid_contents
        start = self.stable_size
        end = start + self.grid_contents_size

        # Wyodrębnienie surowego tensora grid_contents ze spłaszczonego observation_space
        grid_contents = observations[:, start:end].view(
            -1,
            self.grid_contents_shape[0],
            self.grid_contents_shape[1],
            self.grid_contents_shape[2],
        ).permute(0, 3, 1, 2)

        # Extract other inputs
        horse_list_start = end
        horse_list_end = horse_list_start + self.horse_list_size
        horse_list = observations[:, horse_list_start:horse_list_end]

        agent_position_start = horse_list_end
        agent_position_end = agent_position_start + self.agent_position_size
        agent_position = observations[:, agent_position_start:agent_position_end]

        current_horse_index = observations[:, -1].unsqueeze(1)

        # Reshape horse_list to (batch_size, channels=8, num_horses)
        num_horses = self.horse_list_shape[0]  # Number of horses
        horse_list = horse_list.view(-1, 8, num_horses)  # Reshape for 1D CNN
        # Prevent NaNs
        horse_list = th.nan_to_num(horse_list, nan=0.0)

        # Create agent position channel
        batch_size = observations.shape[0]
        agent_channel = th.zeros((batch_size, 1, *self.stable_shape), device=observations.device)
        agent_indices = agent_position.long()
        agent_channel[th.arange(batch_size), 0, agent_indices[:, 0], agent_indices[:, 1]] = 1.0

        # CNN Processing
        combined = th.cat([stable, grid_contents, agent_channel], dim=1)
        cnn_out_combined = th.flatten(self.cnn_combined(combined), start_dim=1)
        cnn_horse_list_out = self.cnn_horse_list(horse_list)
        # MLP Processing
        other_inputs = current_horse_index
        mlp_other_out = self.mlp_other(other_inputs)

        # Combine features
        final_input = th.cat([cnn_out_combined, cnn_horse_list_out, mlp_other_out], dim=1)
        return self.mlp(final_input)


class CNNMLPPolicy(ActorCriticPolicy):
    def __init__(self, *args, original_observation_space: gym.spaces.Dict, **kwargs):
        super().__init__(*args, **kwargs,
                         features_extractor_class=CNNMLPFeatureExtractor,
                         features_extractor_kwargs=dict(features_dim=128,
                                                        original_observation_space=original_observation_space))

