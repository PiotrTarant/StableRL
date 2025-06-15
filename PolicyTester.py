import time
from stable_baselines3 import PPO


from Environment import StableEnvironment
from Functions import read_csv_stables_to_list, read_xls_horses, save_grid_contents_to_excel


def _compute_adjacent_metrics(env):
    """Calculate adjacency metrics for horses placed in the environment."""
    stallion_pairs = 0
    mare_stallion_pairs = 0
    same_surname_pairs = 0

    same_country_pairs = 0
    same_team_pairs = 0
    important_on_2 = 0

    checked = set()

    for position, content in env.grid_contents.items():
        if content["type"] != "horse":
            continue
        horse_data = content["data"]

        if horse_data[7] == 1 and env.original_stable_list[position[0]][position[1]] == 2:
            important_on_2 += 1
        for neighbor in env.get_neighbors(position):
            if neighbor not in env.grid_contents:
                continue
            neighbor_content = env.grid_contents[neighbor]
            if neighbor_content["type"] != "horse":
                continue
            pair = tuple(sorted([position, neighbor]))
            if pair in checked:
                continue
            checked.add(pair)
            neigh_data = neighbor_content["data"]
            if horse_data[5] == "Stallion" and neigh_data[5] == "Stallion":
                stallion_pairs += 1
            if {horse_data[5], neigh_data[5]} == {"Mare", "Stallion"}:
                mare_stallion_pairs += 1
            if horse_data[2] == neigh_data[2]:
                same_surname_pairs += 1
            if horse_data[4] == neigh_data[4]:
                same_country_pairs += 1
            if horse_data[6] == neigh_data[6]:
                same_team_pairs += 1

    return (
        stallion_pairs,
        mare_stallion_pairs,
        same_surname_pairs,
        same_country_pairs,
        same_team_pairs,
        important_on_2,
    )





def test_policy(model_path: str, stable_csv: str, horse_xls: str):
    """Run a trained policy and print evaluation metrics."""
    stable_list = read_csv_stables_to_list(stable_csv)
    horse_list = read_xls_horses(horse_xls)

    env = StableEnvironment(stable_list, horse_list)
    model = PPO.load(model_path)

    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    start = time.time()

    while not done:

        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += float(reward)

    duration = time.time() - start

    (
        stallions,
        mare_stallion,
        same_surname,
        same_country,
        same_team,
        important_on_2,
    ) = _compute_adjacent_metrics(env)

    save_grid_contents_to_excel(env.grid_contents)

    print(f"Czas wykonania: {duration:.2f}s")
    print(f"Łączna nagroda: {total_reward:.2f}")
    print(f"Ogiery obok siebie: {stallions}")
    print(f"Klacze obok ogierów: {mare_stallion}")
    print(f"Konie o tym samym nazwisku obok siebie: {same_surname}")
    print(f"Zawodnicy z tego samego kraju obok siebie: {same_country}")
    print(f"Zawodnicy z tego samego zespołu obok siebie: {same_team}")
    print(f"Ważni zawodnicy na polu 2: {important_on_2}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test trained PPO policy")
    parser.add_argument("--model", required=True, help="Path to the trained model")
    parser.add_argument("--stable", default="testowa_stajnia.csv", help="CSV file with stable layout")
    parser.add_argument("--horses", default="test_lista_koni_40.xls", help="Excel file with horse list")
    args = parser.parse_args()

    test_policy(args.model, args.stable, args.horses)
