from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch as th
from CNNMLPPolicy import CNNMLPPolicy
from Environment import StableEnvironment



def train_model_ppo(stable_list, horse_list, policy, policy_path=None):
    # Disable internal normalization to use raw observations and rewards
    env = DummyVecEnv([
        lambda: StableEnvironment(stable_list, horse_list, normalize_output=False)
    ])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # last change ent_coef 0.2-> 0.1 vf_coef 1.5 -> 1
    # Tworzenie algorytmu PPO z hybrydową siecią CNN-MLP
    model = PPO(policy, env, verbose=1,
                device='cuda', ent_coef=0.1, clip_range=0.2, vf_coef=1, gamma=0.98,
                learning_rate=0.0003,
                policy_kwargs=dict(original_observation_space=env.original_observation_space),
                normalize_advantage=True,
                tensorboard_log="./ppo_tensorboard/")

    #model.policy.load_state_dict(PPO.load(policy_path, env=env).policy.state_dict())
    #print(f"Loaded policy from {policy_path}")

    # Trening modelu
    iteration = 0
    while True:
        iteration += 1
        model.learn(total_timesteps=50000, tb_log_name="run_1")
        model.save(f"model_PPO/stable_environment_CNNMLP_{iteration}")
        env.save("model_PPO/vec_normalize.pkl")
