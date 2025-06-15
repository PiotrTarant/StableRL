import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np




def read_xls_horses(file_path):
    # Wczytaj dane z pliku Excel (.xls), ograniczając do wybranych kolumn
    selected_columns = [1, 2, 5, 6, 7, 11, 22, 23]
    df = pd.read_excel(file_path, engine='xlrd', usecols=selected_columns)

    # Konwertuj dane z DataFrame na listę
    # data = df.values.tolist()

    return df

def read_csv_stables_to_list(file_path):
    df = pd.read_csv(file_path)


    #data = df.values.tolist()

    # Sprawdzenie, czy wszystkie elementy są liczbami całkowitymi
    processed_data = []
    for row in df.values:
        processed_row = []
        for cell in row:
            if isinstance(cell, str) and ";" in cell:
                # Rozdziel dane, jeśli są zapisane w jednej komórce jako "1;3;1;4"
                split_values = map(int, cell.split(";"))
                processed_row.extend(split_values)
            else:
                # Jeśli nie trzeba rozdzielać, dodaj wartość jako liczbę
                processed_row.append(int(cell))
        processed_data.append(processed_row)

    return processed_data

def save_grid_contents_to_excel(grid_contents, file_name="grid_contents.xlsx"):
    # Ustal maksymalne wymiary siatki (grid), aby utworzyć odpowiednią liczbę wierszy i kolumn
    max_row = max(pos[0] for pos in grid_contents.keys()) + 1  # Wiersze
    max_col = max(pos[1] for pos in grid_contents.keys()) + 1  # Kolumny

    # Tworzenie DataFrame z pustymi wartościami
    grid_data = pd.DataFrame([['' for _ in range(max_col)] for _ in range(max_row)])

    # Wypełnianie DataFrame na podstawie `grid_contents`
    for (row, col), horse in grid_contents.items():
        grid_data.iat[row, col] = horse  # Wstawienie konia w odpowiednie pole

    # Zapis DataFrame do pliku Excel
    grid_data.to_excel(file_name, index=False, header=False, engine='openpyxl')
    print(f"Zawartość grid_contents została zapisana do pliku: {file_name}")

def encode_horse_list(horse_list):
    # Inicjalizujemy LabelEncodery dla stringów
    name_encoder = LabelEncoder()
    surname_encoder = LabelEncoder()
    country_encoder = LabelEncoder()
    horse_name_encoder = LabelEncoder()
    gender_mapping = {"Mare": 0, "Gelding": 1, "Stallion": 2}

    # Tworzenie pełnej kopii horse_list
    encoded_horse_list = horse_list.copy()

    # Kodowanie wszystkich kolumn stringów
    encoded_horse_list['Nazwisko'] = surname_encoder.fit_transform(encoded_horse_list['Nazwisko'])
    encoded_horse_list['Imię'] = name_encoder.fit_transform(encoded_horse_list['Imię'])
    encoded_horse_list['Kraj (Zawodnik)'] = country_encoder.fit_transform(encoded_horse_list['Kraj (Zawodnik)'])
    encoded_horse_list['Nazwa'] = horse_name_encoder.fit_transform(encoded_horse_list['Nazwa'])

    # Płeć (różne typy koni)
    encoded_horse_list['Płeć'] = encoded_horse_list['Płeć'].map(gender_mapping)

    # Normalizacja `Team` i inne cechy numeryczne
    encoded_horse_list['Team'] = encoded_horse_list['Team'] / encoded_horse_list['Team'].max()
    encoded_horse_list['Numer konia'] = encoded_horse_list['Numer konia'] / encoded_horse_list['Numer konia'].max()

    return encoded_horse_list


def encode_grid_contents(grid_contents, grid_array=None, grid_size=(10, 10), max_horse_number=400, max_team=15, encoders=None):
    """
    Funkcja kodująca dane grid_contents do tablicy NumPy z uwzględnieniem unikalnego numeru konia.

    :param grid_array:
    :param grid_contents: słownik, gdzie klucze to współrzędne (x, y),
                          a wartości to dane koni i zawodników.
    :param grid_size: Rozmiar siatki (grid_x, grid_y), np. (10, 10)
    :param max_horse_number: Maksymalny numer konia (normalizacja).
    :param max_team: Maksymalny numer drużyny (normalizacja).
    :return: 3-wymiarowa tablica NumPy reprezentująca zakodowaną siatkę.
    """
    # Przygotowanie pustej tablicy o wymiarach (grid_x, grid_y, num_features)
    if grid_array is None:
        num_features = 8  # Liczba cech: numer_konia, horse_id, player_id, gender, team, important
        grid_array = np.zeros((grid_size[0], grid_size[1], num_features), dtype=np.float32)

    if encoders is None:
        raise ValueError("Encoders dictionary is required")

    name_encoder = encoders["name"]
    surname_encoder = encoders["surname"]
    country_encoder = encoders["country"]
    horse_name_encoder = encoders["horse_name"]
    gender_mapping = {"Mare": 0, "Gelding": 1, "Stallion": 2}


    for (x, y), content in grid_contents.items():
        if content["type"] == "horse":  # Sprawdzamy typ obiektu (np. koń)
            horse_data = content["data"]

            grid_array[x, y, 0] = horse_data[0] / max_horse_number
            grid_array[x, y, 1] = horse_name_encoder.transform([horse_data[1]])
            grid_array[x, y, 2] = surname_encoder.transform([horse_data[2]])
            grid_array[x, y, 3] = name_encoder.transform([horse_data[3]])
            grid_array[x, y, 4] = country_encoder.transform([horse_data[4]])
            grid_array[x, y, 5] = gender_mapping[horse_data[5]]
            grid_array[x, y, 6] = horse_data[6] / max_team
            grid_array[x, y, 7] = 1 if horse_data[7]==1 else 0
        elif content["type"] == "healing_box":
            grid_array[x, y, 5] = 3
        elif content["type"] == "antidoping_box":
            grid_array[x, y, 5] = 4

    return grid_array
