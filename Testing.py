from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from Environment import StableEnvironment
from Functions import save_grid_contents_to_excel

def test_model_ppo(stable_list, horse_list, model_name):
    # Załaduj zapisany model
    env = DummyVecEnv([lambda: StableEnvironment(stable_list, horse_list, normalize_output=False)])
    env = VecNormalize.load("model_PPO/vec_normalize.pkl", env)
    model = PPO.load(model_name, env=env)

    # Przetestuj agenta w środowisku
    obs, info = env.reset()
    done = False

    print("\nTest działania modelu PPO:")
    while not done:
        # Wybieranie akcji przez model
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, done, truncated, info = env.step(action)

        # Wyświetlenie aktualnego stanu, akcji i nagrody
        print(f"Stan: {obs}, Akcja: {action}, Nagroda: {reward}")

    print("\nKonie i ich pozycje w grid_contents:")
    for position, horse in env.grid_contents.items():
        print(f"Pole: {position}, Koń: {horse}")
    save_grid_contents_to_excel(env.grid_contents)

