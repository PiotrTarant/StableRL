from stable_baselines3.ppo import MlpPolicy

import Functions
from Extractor import HybridCNNMLPPolicy
from Show_env import show_env
from Testing import test_model_ppo
from Train import train_model_ppo
from Environment import StableEnvironment
from GNNExtractor import GNNCNNPolicy
from CNNMLPPolicy import CNNMLPPolicy

horse_file_path = "test_lista_koni_20.xls"

horse_list = Functions.read_xls_horses(horse_file_path)

stable_file_path = "testowa_stajnia.csv"

stable_list = Functions.read_csv_stables_to_list(stable_file_path)

env = StableEnvironment(stable_list, horse_list)
show_env(env, horse_list, stable_list)

train_model_ppo(stable_list, horse_list, CNNMLPPolicy)

#model_name = "model_PPO/stable_environment_gnn_4.zip"
#test_model_ppo(stable_list, horse_list, model_name)

