from stable_baselines3.ppo import MlpPolicy

import Functions
from Extractor import HybridCNNMLPPolicy
from Show_env import show_env
from Testing import test_model_ppo
from Train import train_model_ppo
from Environment import StableEnvironment
from GNNExtractor import GNNCNNPolicy
from CNNMLPPolicy import CNNMLPPolicy

horse_file_path = "ZG_lista_koni.xls"

horse_list = Functions.read_xls_horses(horse_file_path)

stable_file_path = "plan_stajni_zg.csv"

stable_list = Functions.read_csv_stables_to_list(stable_file_path)

env = StableEnvironment(stable_list, horse_list)
show_env(env, horse_list, stable_list)

train_model_ppo(stable_list, horse_list, CNNMLPPolicy, policy_path="stable_environment_cnn_mlp_zg_200k.zip")

model_name = "model_PPO/stable_environment_gnn_2.zip"
test_model_ppo(stable_list, horse_list, model_name)

