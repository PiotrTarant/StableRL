import torch as th
import gymnasium as gym
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.ao.nn.quantized import LayerNorm
from torch_geometric.nn.glob import global_mean_pool
from torch_geometric.nn.pool import radius_graph, knn_graph
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.conv import GCNConv, SAGEConv





class GNNFeatureExtractor(BaseFeaturesExtractor):
    """Graph Neural Network Feature Extractor"""

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        stable_shape = observation_space["stable"].shape
        stable_size = np.prod(stable_shape)  # Ilość boksów jako węzły w grafie
        agent_pos_size = observation_space["agent_position"].shape[0]  # (x, y)
        #self.projection = th.nn.Linear(features_dim, features_dim)
        self.projection = th.nn.Sequential(
            th.nn.Linear(features_dim, features_dim),
            th.nn.LayerNorm(features_dim),  # Normalizacja
            th.nn.ReLU()  # Aktywacja
        )




        # Warstwy GNN
        self.conv1 = GCNConv(1, 64)  # 1 wejściowy kanał (cecha boksu), 64 wyjściowe
        self.norm1 = th.nn.LayerNorm(64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, features_dim) # Druga warstwa GNN
        self.norm2 = th.nn.LayerNorm(features_dim)
        self.dropout = th.nn.Dropout(0.1)

    def forward(self, observations: dict) -> th.Tensor:
        stable = observations["stable"].view(observations["stable"].shape[0], -1)
        agent_position = observations["agent_position"]

        batch_size = stable.shape[0]

        node_features = stable.unsqueeze(-1).float()  # (batch_size, num_nodes, 1)


        # Tworzenie grafu
        edge_list = []
        num_nodes = stable.shape[1]  # Liczba wierzchołków w stable
        sqrt_nodes = int(num_nodes ** 0.5)  # Przyjmujemy kwadratową siatkę

        for i in range(num_nodes):
            if i % sqrt_nodes != 0:
                edge_list.append((i, i - 1))
            if (i + 1) % sqrt_nodes != 0:
                edge_list.append((i, i + 1))
            if i >= sqrt_nodes:
                edge_list.append((i, i - sqrt_nodes))
            if i < num_nodes - sqrt_nodes:
                edge_list.append((i, i + sqrt_nodes))

        # Walidacja krawędzi
        #edge_list = [(src, dst) for src, dst in edge_list if src < num_nodes and dst < num_nodes]
        edge_index = knn_graph(node_features.squeeze(-1), k=4, loop=True)
        #edge_index = th.tensor(edge_list, dtype=th.long).t().contiguous()



        # Przepuszczenie przez GNN
        x = self.conv1(node_features, edge_index).relu()
        x = self.norm1(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.norm2(x)

        #graph_embedding = th.max(x, dim=1).values
        graph_embedding = global_mean_pool(x, batch=None)
        projected_output = self.projection(graph_embedding)

        return projected_output



from stable_baselines3.common.policies import ActorCriticPolicy

class GNNCNNPolicy(ActorCriticPolicy):
    """Polityka PPO z niestandardowym ekstraktorem GNN"""
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=GNNFeatureExtractor,  # Nasz ekstraktor GNN
            features_extractor_kwargs=dict(features_dim=128),  # Rozmiar wyjściowy cech
        )




