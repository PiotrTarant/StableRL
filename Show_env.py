import pandas as pd

def show_env(env, horse_list, stable_list):
    # Ustawienia, by wyświetlić wszystkie wiersze i kolumny
    pd.set_option('display.max_rows', None)  # Wyświetl wszystkie wiersze
    pd.set_option('display.max_columns', None)  # Wyświetl wszystkie kolumny
    pd.set_option('display.width', None)  # Dopasowanie szerokości terminala
    pd.set_option('display.max_colwidth', None)  # Wyświetl pełne wartości w komórkach (bez ich skracania)

    print(horse_list)

    print(stable_list)


    print("Przed resetem:", env.stable_list)
    env.stable_list[0][0] = -999  # Wprowadź zmiany testowe
    print("Po modyfikacji:", env.stable_list)
    env.reset()
    print("Po resecie:", env.stable_list)

    # Zresetowanie środowiska i wyświetlenie stanu początkowego
    observation = env.reset()
    print("Stan początkowy:")
    print(observation)

    # Symulacja kroków agenta
    actions = [
        5,
        1,
        4,
        0

    ]

    print("\nSymulacja akcji:")
    for action in actions:
        observation, reward, done, truncated, info = env.step(action)
        print(f"Akcja: {action}, Obserwacja: {observation}, Nagroda: {reward}, Zakończono: {done}")

    # Wyświetlenie zawartości słownika grid_contents (gdzie i co zostało umieszczone)
    print("\nZawartość stajni po symulacji:")
    print(env.grid_contents)