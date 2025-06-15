from MaskablePPO import MaskablePPO
import torch as th
from CNNMLPPolicy import CNNMLPPolicy
from Environment import StableEnvironment
from ActionMaskWrapper import ActionMaskWrapper


def train_model_ppo(stable_list, horse_list, policy, policy_path=None):
    # Disable internal normalization to use raw observations and rewards
    base_env = StableEnvironment(stable_list, horse_list)
    env = ActionMaskWrapper(base_env)

    # last change ent_coef 0.2-> 0.1 vf_coef 1.5 -> 1
    # Tworzenie algorytmu PPO z hybrydową siecią CNN-MLP
    model = MaskablePPO(policy, env, verbose=1,
                device='cuda', ent_coef=0.1, clip_range=0.2, vf_coef=1, gamma=0.98,
                learning_rate=0.0003,
                policy_kwargs=dict(original_observation_space=base_env.original_observation_space),
                normalize_advantage=True,
                tensorboard_log="./ppo_tensorboard/")

    #model.policy.load_state_dict(MaskablePPO.load(policy_path, env=env).policy.state_dict())
    #print(f"Loaded policy from {policy_path}")

    # Trening modelu
    iteration = 0
    while True:
        iteration += 1
        model.learn(total_timesteps=50000, tb_log_name="run_1")
        model.save(f"model_PPO/stable_environment_CNNMLP_{iteration}")
