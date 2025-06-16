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


def evaluate_policy(model_path: str, stable_csv: str, horse_xls: str, runs: int = 1000):
    """Run a trained policy multiple times and report average metrics and reward."""
    stable_list = read_csv_stables_to_list(stable_csv)
    horse_list = read_xls_horses(horse_xls)

    model = PPO.load(model_path)

    metrics_sum = [0, 0, 0, 0, 0, 0]
    reward_sum = 0.0

    env = StableEnvironment(stable_list, horse_list)
    for _ in range(runs):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += float(reward)

        reward_sum += total_reward
        metrics = _compute_adjacent_metrics(env)
        metrics_sum = [m + v for m, v in zip(metrics_sum, metrics)]

    avg_metrics = [m / runs for m in metrics_sum]
    avg_reward = reward_sum / runs

    print(f"Średnia łączna nagroda: {avg_reward:.2f}")
    print(f"Średnia ogierów obok siebie: {avg_metrics[0]:.2f}")
    print(f"Średnia klaczy obok ogierów: {avg_metrics[1]:.2f}")
    print(f"Średnia koni o tym samym nazwisku obok siebie: {avg_metrics[2]:.2f}")
    print(f"Średnia zawodników z tego samego kraju obok siebie: {avg_metrics[3]:.2f}")
    print(f"Średnia zawodników z tego samego zespołu obok siebie: {avg_metrics[4]:.2f}")
    print(f"Średnia ważnych zawodników na polu 2: {avg_metrics[5]:.2f}")

    return avg_metrics, avg_reward


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test trained PPO policy")
    parser.add_argument("--model", required=True, help="Path to the trained model")
    parser.add_argument("--stable", default="testowa_stajnia.csv", help="CSV file with stable layout")
    parser.add_argument("--horses", default="test_lista_koni_40.xls", help="Excel file with horse list")
    parser.add_argument("--evaluate", action="store_true", help="Run multiple evaluations")
    parser.add_argument("--runs", type=int, default=1000, help="Number of evaluation runs")
    args = parser.parse_args()

    if args.evaluate:
        evaluate_policy(args.model, args.stable, args.horses, args.runs)
    else:
        test_policy(args.model, args.stable, args.horses)
