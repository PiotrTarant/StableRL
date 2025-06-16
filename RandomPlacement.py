import random
from typing import Tuple

import numpy as np

from Environment import StableEnvironment
from Functions import (
    read_csv_stables_to_list,
    read_xls_horses,
    save_grid_contents_to_excel,
)


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


def _random_placement_once(
    stable_csv: str, horse_xls: str, save_excel: bool = False
) -> Tuple[Tuple[int, int, int, int, int, int], float]:
    """Place horses randomly once and return metrics and total reward."""
    stable_list = read_csv_stables_to_list(stable_csv)
    horse_list = read_xls_horses(horse_xls)

    env = StableEnvironment(stable_list, horse_list)
    env.reset()

    # Ignore special boxes so that episode can end after placing horses
    env.healing_boxes_remaining = 0
    env.antidoping_boxes_remaining = 0

    available = [
        (i, j)
        for i in range(env.grid_size[0])
        for j in range(env.grid_size[1])
        if env.original_stable_list[i][j] in [1, 2]
    ]
    random.shuffle(available)

    total_reward = 0.0
    for horse_data in env.horse_list:
        if not available:
            break
        position = available.pop()
        env.agent_position = np.array(position, dtype=np.int32)
        _, reward, _, _, _ = env.step(4)
        total_reward += float(reward)

    if save_excel:
        save_grid_contents_to_excel(env.grid_contents, "grid_contents_random.xlsx")

    metrics = _compute_adjacent_metrics(env)
    return metrics, total_reward


def random_placement(stable_csv: str, horse_xls: str):
    """Randomly place horses and print metrics."""
    metrics, total_reward = _random_placement_once(stable_csv, horse_xls, True)

    (
        stallions,
        mare_stallion,
        same_surname,
        same_country,
        same_team,
        important_on_2,
    ) = metrics

    print(f"Łączna nagroda: {total_reward:.2f}")
    print(f"Ogiery obok siebie: {stallions}")
    print(f"Klacze obok ogierów: {mare_stallion}")
    print(f"Konie o tym samym nazwisku obok siebie: {same_surname}")
    print(f"Zawodnicy z tego samego kraju obok siebie: {same_country}")
    print(f"Zawodnicy z tego samego zespołu obok siebie: {same_team}")
    print(f"Ważni zawodnicy na polu 2: {important_on_2}")


def random_placement_multiple(
    stable_csv: str, horse_xls: str, trials: int = 1000
) -> Tuple[Tuple[float, float, float, float, float, float], float]:
    """Run random placement several times and return average metrics and reward."""
    metrics_sum = np.zeros(6, dtype=np.float64)
    reward_sum = 0.0

    for _ in range(trials):
        metrics, total_reward = _random_placement_once(stable_csv, horse_xls)
        metrics_sum += np.array(metrics, dtype=np.float64)
        reward_sum += total_reward

    avg_metrics = tuple(metrics_sum / trials)
    avg_reward = reward_sum / trials
    return avg_metrics, avg_reward


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Random horse placement")
    parser.add_argument("--stable", default="testowa_stajnia.csv", help="CSV file with stable layout")
    parser.add_argument("--horses", default="test_lista_koni_20.xls", help="Excel file with horse list")
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of random placements to average over",
    )
    args = parser.parse_args()

    if args.trials > 1:
        avg_metrics, avg_reward = random_placement_multiple(
            args.stable, args.horses, args.trials
        )
        (
            stallions,
            mare_stallion,
            same_surname,
            same_country,
            same_team,
            important_on_2,
        ) = avg_metrics
        print(f"Średnia łączna nagroda: {avg_reward:.2f}")
        print(f"Średnie ogiery obok siebie: {stallions:.2f}")
        print(f"Średnie klacze obok ogierów: {mare_stallion:.2f}")
        print(f"Średnie konie o tym samym nazwisku: {same_surname:.2f}")
        print(f"Średnie pary z tego samego kraju: {same_country:.2f}")
        print(f"Średnie pary z tego samego zespołu: {same_team:.2f}")
        print(f"Średnio ważni zawodnicy na polu 2: {important_on_2:.2f}")
    else:
        random_placement(args.stable, args.horses)
