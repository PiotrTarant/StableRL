import gymnasium as gym
import numpy as np
import math

from gymnasium.spaces import flatten_space, flatten

from Functions import encode_horse_list, encode_grid_contents


class StableEnvironment(gym.Env):
    def __init__(self, stable_list, horse_list, normalize_output=True):
        """
        Inicjalizacja środowiska.
        :param stable_list: Lista pól stajni (processed_data, wartości 1-10).
        :param horse_list: Data Frame z listą koni.
        """
        super(StableEnvironment, self).__init__()

        self.normalize_output = normalize_output

        self.current_horse_index = 0

        self.stable_list = np.array(stable_list)  # Stajnia jako siatka 2D
        self.original_stable_list = np.array([row.copy() for row in self.stable_list])
        self.horse_list = horse_list.values.tolist()  # Lista koni
        self.horses_remaining = len(self.horse_list)  # Liczba koni do umieszczenia
        self.healing_boxes_remaining = 2  # Liczba boksów leczących do umieszczenia
        self.antidoping_boxes_remaining = 2  # Liczba boksów antydopingowych do umieszczenia
        self.grid_size = self.stable_list.shape  # Rozmiar stajni (np. 10x10)

        self.fields_5 = []  # Lista pól o wartości 5
        self.fields_6 = []  # Lista pól o wartości 6
        self.fields_7 = []  # Lista pól o wartości 7

        # Wyszukiwanie współrzędnych dla wartościowych pól
        for i in range(self.stable_list.shape[0]):  # Iteracja po wierszach
            for j in range(self.stable_list.shape[1]):  # Iteracja po kolumnach
                if self.stable_list[i, j] == 5:
                    self.fields_5.append((i, j))
                elif self.stable_list[i, j] == 6:
                    self.fields_6.append((i, j))
                elif self.stable_list[i, j] == 7:
                    self.fields_7.append((i, j))

        # Informacje o polach: klucz -> (x, y), wartość -> zawartość pola (koń, box, itp.)
        self.grid_contents = {}  # {(x, y): {"type": "horse", "data": {...}}, ...}

        # Współrzędne agenta (start na polu w lewym górnym rogu)
        self.agent_position = [0, 0]

        # Tablica zajętych miejsc
        self.occupied_positions = np.zeros_like(self.stable_list, dtype=bool)

        # Akcje agenta
        self.action_space = gym.spaces.Discrete(7)

        #Kodowanie horse_list oraz grid_contents
        self.max_horse_id = max(self.horse_list, key=lambda x: x[0])[0]
        self.max_horse_team = max(self.horse_list, key=lambda x: x[6])[6]
        self.encoded_horse_list = encode_horse_list(horse_list)
        self.encoded_grid_contents = np.zeros((*self.grid_size, 8), dtype=np.float32)

        # Stan - informacje o stajni i pozycji agenta
        self.original_observation_space = gym.spaces.Dict({
            "stable": gym.spaces.Box(low=1, high=10, shape=self.stable_list.shape, dtype=np.int32),
            "agent_position": gym.spaces.Box(low=0, high=max(self.grid_size), shape=(2,), dtype=np.int32),
            "horse_list": gym.spaces.Box(low=0, high=1, shape=self.encoded_horse_list.shape, dtype=np.float32),
            "grid_contents": gym.spaces.Box(low=0, high=1, shape=self.encoded_grid_contents.shape, dtype=np.float32),
            "current_horse_index": gym.spaces.Box(low=0, high=len(self.horse_list), shape=(1,), dtype=np.int32),
        })

        self.observation_space = flatten_space(self.original_observation_space)

        # Statystyki normalizacji
        self.obs_mean = np.zeros(self.observation_space.shape[0])
        self.obs_std = np.ones(self.observation_space.shape[0])
        self.reward_mean = 0
        self.reward_std = 1
        # Parametry normalizacji
        self.normalization_alpha = 0.01

    def normalize_observation(self, obs):
        """Normalizacja spłaszczonej obserwacji."""
        return (obs - self.obs_mean) / (self.obs_std + 1e-8)

    def update_normalization_stats(self, obs):
        """Zaktualizuj statystyki normalizacji na podstawie nowej obserwacji."""
        self.obs_mean = (1 - self.normalization_alpha) * self.obs_mean + self.normalization_alpha * obs
        self.obs_std = np.sqrt(
            (1 - self.normalization_alpha) * np.square(self.obs_std) +
            self.normalization_alpha * np.square(obs - self.obs_mean)
        )

    def normalize_reward(self, reward):
        """Normalizacja nagrody."""
        return (reward - self.reward_mean) / (self.reward_std + 1e-8)

    def update_reward_stats(self, reward):
        """Zaktualizuj statystyki normalizacji nagrody."""
        self.reward_mean = (1 - self.normalization_alpha) * self.reward_mean + self.normalization_alpha * reward
        self.reward_std = np.sqrt(
            (1 - self.normalization_alpha) * np.square(self.reward_std) +
            self.normalization_alpha * np.square(reward - self.reward_mean)
        )

    def place_horse(self, position, horse_data):
        """
        Umieszcza konia na wskazanym polu w stajni.
        :param position: Tuple (x, y), współrzędne pola.
        :param horse_data: Dane konia (np. pojedynczy rekord z horse_list).
        """
        if position in self.grid_contents:
            raise ValueError(f"Pole {position} jest już zajęte przez {self.grid_contents[position]}")
        self.grid_contents[position] = {"type": "horse", "data": horse_data}
        print(f"Koń {horse_data} umieszczony na polu {position}.")

    def place_healing_box(self, position):
        """
        Umieszcza boks leczący na wskazanym polu.
        :param position: Tuple (x, y), współrzędne pola.
        """
        if position in self.grid_contents:
            raise ValueError(f"Pole {position} jest już zajęte przez {self.grid_contents[position]}")
        self.grid_contents[position] = {"type": "healing_box"}
        self.healing_boxes_remaining -= 1


        print(f"Boks leczący umieszczony na polu {position}.")

    def place_antidoping_box(self, position):
        """
        Umieszcza boks antydopingowy na wskazanym polu.
        :param position: Tuple (x, y), współrzędne pola.
        """
        if position in self.grid_contents:
            raise ValueError(f"Pole {position} jest już zajęte przez {self.grid_contents[position]}")
        self.grid_contents[position] = {"type": "antidoping_box"}
        self.antidoping_boxes_remaining -= 1


        print(f"Boks antydopingowy umieszczony na polu {position}.")

    def get_neighbors(self, position):
        """
        Pobiera pośrednich sąsiadów z uwzględnieniem korytarza (3).
        :param position: Tuple (x, y) - współrzędne pola.
        :return: Lista współrzędnych pośrednio sąsiadujących pól.
        """
        x, y = position
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            # Sprawdzenie, czy współrzędne są w granicach siatki
            if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                if self.stable_list[nx][ny] == 3:  # Jeśli między jest pole 3, sprawdź kolejne
                    extended_x = nx + dx
                    extended_y = ny + dy
                    if (0 <= extended_x < self.grid_size[0] and
                            0 <= extended_y < self.grid_size[1]):
                        neighbors.append((extended_x, extended_y))
                else:
                    neighbors.append((nx, ny))

        return neighbors

    def get_direct_neighbors(self, position):
        """
        Pobiera bezpośrednich sąsiadów.
        :param position: Tuple (x, y) - współrzędne pola.
        :return: Lista współrzędnych bezpośrednio sąsiadujących pól.
        """
        x, y = position
        neighbors = []

        # Definicja bezpośrednich sąsiadów (góra, dół, lewo, prawo)
        direct_neighbors = [
            (x - 1, y),  # Góra
            (x + 1, y),  # Dół
            (x, y - 1),  # Lewo
            (x, y + 1)  # Prawo
        ]

        # Dodanie sąsiadów tylko w zakresie planszy
        for nx, ny in direct_neighbors:
            if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                neighbors.append((nx, ny))

        return neighbors

    def calculate_min_distance(self, position, fields) -> float:
        """
        Oblicz minimalny dystans Euklidesowy między `position` a listą `fields`.
        """
        x, y = position
        min_distance = float('inf')  # Początkowo nieskończoność

        for field_x, field_y in fields:
            distance = math.sqrt((x - field_x) ** 2 + (y - field_y) ** 2)  # Odległość Euklidesowa
            if distance < min_distance:
                min_distance = distance

        return min_distance

    def find_closest_target(self, start_position):
        from queue import Queue
        visited = set()
        q = Queue()
        q.put(start_position)

        while not q.empty():
            current_x, current_y = q.get()
            # Jeżeli napotkamy pole 1 lub 2, zwracamy jego współrzędne
            if self.stable_list[current_x][current_y] in [1, 2]:
                return current_x, current_y

            # Przechodzimy po sąsiadach
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor_x, neighbor_y = current_x + dx, current_y + dy
                if (0 <= neighbor_x < self.stable_list.shape[0] and
                        0 <= neighbor_y < self.stable_list.shape[1] and
                        (neighbor_x, neighbor_y) not in visited):
                    visited.add((neighbor_x, neighbor_y))
                    q.put((neighbor_x, neighbor_y))

        # Jeżeli żadne pole 1 lub 2 nie jest dostępne
        return start_position



    def reset(self, **kwargs):
        """Resetowanie środowiska.
        :param **kwargs:
        """
        self.current_horse_index = 0
        self.horses_remaining = len(self.horse_list)
        self.healing_boxes_remaining = 2
        self.antidoping_boxes_remaining = 2
        self.agent_position =np.array([0, 0], dtype=np.int32)
        self.occupied_positions = np.zeros_like(self.stable_list, dtype=bool)
        self.stable_list = np.array([row.copy() for row in self.original_stable_list])
        self.grid_contents = {}
        self.encoded_grid_contents = np.zeros((*self.grid_size, 8), dtype=np.float32)

        observation = {
            "stable": self.stable_list.astype(np.int32),  # Dopasuj typ danych
            "agent_position": np.array(self.agent_position, dtype=np.int32),
            "horse_list": self.encoded_horse_list.to_numpy().astype(np.float32).reshape(
                self.original_observation_space["horse_list"].shape),
            "grid_contents": self.encoded_grid_contents.astype(np.float32).reshape(
                self.original_observation_space["grid_contents"].shape),
            "current_horse_index": np.array([self.current_horse_index], dtype=np.int32),
        }

        print(f"Encoded grid_contents shape: {self.encoded_grid_contents.shape}")
        print(f"Original observation_space shape: {self.original_observation_space['grid_contents'].shape}")

        flat_observation = flatten(self.original_observation_space, observation)

        if self.normalize_output:
            self.update_normalization_stats(flat_observation)
            flat_observation = self.normalize_observation(flat_observation)

        return flat_observation, {}


    def step(self, action):
        """
        Wykonanie akcji.
        :param action: Numer akcji (0-7).
        :return: observation, reward, done, info
        """

        if isinstance(action, np.ndarray):
            action = action.item()  # Konwertowanie na int
        print("Received action:", action, "Type:", type(action))
        x, y = self.agent_position
        reward = 0
        done = False
        if action in [4, 5, 6] and self.stable_list[x][y] not in [1, 2]:
            print("Akcja niedozwolona – agent nie może wykonać tej akcji na tym polu!")
            reward -= 3
            if y > 0:  # Lewo
                action = 2
            elif y < self.grid_size[1] - 1:  # Prawo
                action = 3



        # Lista akcji (0: dół, 1: góra, 2: lewo, 3: prawo, 4: umieść konia, 5: umieść boks leczący, 6: umieść boks antydopingowy)
        # Ruch agenta
        if action in [0, 1, 2, 3]:
            # poruszanie sie jedynie po polach 1 i 2 dlaczego wczesniej na to nie wpadłem xd
            directions = {
                0: (-1, 0),  # góra
                1: (1, 0),  # dół
                2: (0, -1),  # lewo
                3: (0, 1),  # prawo
            }

            dx, dy = directions.get(action, (0, 0))
            new_x, new_y = x + dx, y + dy
            while True:
                if not (0 <= new_x < self.stable_list.shape[0] and 0 <= new_y < self.stable_list.shape[1]):
                    new_x, new_y = x, y
                    self.agent_position = np.array([new_x, new_y], dtype=np.int32)
                    break
                if self.stable_list[new_x, new_y] in [1, 2]:
                    self.agent_position = np.array([new_x, new_y], dtype=np.int32)
                    break
                new_x, new_y = new_x + dx, new_y + dy
            reward -= 0.05  # Kara za ruch


        # Umieszczenie konia
        elif action == 4:
            if self.horses_remaining > 0 and (x, y) not in self.grid_contents:
                # Pobieramy dane nowo umieszczonego konia
                horse_data = self.horse_list[self.current_horse_index]
                encoded_horse_data = self.encoded_horse_list.iloc[self.current_horse_index]
                self.current_horse_index += 1

                # Zamykamy pole
                self.stable_list[x][y] = 4



                # Umieszczenie konia na planszy
                self.place_horse((x, y), horse_data)
                self.horses_remaining -= 1
                self.encoded_grid_contents = encode_grid_contents(self.grid_contents, self.encoded_grid_contents, self.grid_size, self.max_horse_id, self.max_horse_team)


                reward += 1  # Standardowa nagroda za umieszczenie konia

                # Pobierz nazwisko zawodnika, imię zawodnika i zespół konia
                current_last_name = horse_data[2]  # Nazwisko
                current_first_name = horse_data[3]  # Imię
                current_team = horse_data[6]  # Zespół
                current_country = horse_data[4]  # Kraj pochodzenia
                current_gender = horse_data[5]  # Płeć (Stallion, Mare, Gelding)

                # Sprawdzamy sąsiadów tego pola za pomocą funkcji get_neighbors
                neighbors = self.get_neighbors((x, y))

                # Sprawdzamy bezpośrednich sąsiadów (pomijając pole o wartości 3)
                direct_neighbors = self.get_direct_neighbors((x, y))


                # Nagroda za bliskie pola specjalne
                if horse_data[7] == 1:
                    # Nagroda za umieszczenie ważnego konia w stajni murowanej
                    if self.stable_list[x][y] == 2:
                        reward += 2
                    # Nagrody podstawowe
                    base_reward_5 = 1
                    base_reward_6 = 1
                    base_reward_7 = 1

                    # Minimalne odległości do pól wartościowych
                    min_distance_5 = self.calculate_min_distance((x, y), self.fields_5)
                    min_distance_6 = self.calculate_min_distance((x, y), self.fields_6)
                    min_distance_7 = self.calculate_min_distance((x, y), self.fields_7)

                    # Dolicz nagrodę za bycie blisko pól
                    if min_distance_5 > 0:  # Unikamy dzielenia przez zero
                        reward += (base_reward_5 / min_distance_5)  # Nagroda za bliskość do pól o wartości 5
                    if min_distance_6 > 0:
                        reward += (base_reward_6 / min_distance_6)  # Nagroda za bliskość do pól o wartości 6
                    if min_distance_7 > 0:
                        reward += (base_reward_7 / min_distance_7)  # Nagroda za bliskość do pól o wartości 7

                # Nagrody
                for nx, ny in neighbors:
                    if (nx, ny) in self.grid_contents:
                        neighbor = self.grid_contents[(nx, ny)]
                        if neighbor["type"] == "horse":
                            neighbor_data = neighbor["data"]
                            # Sprawdzenie warunków-tego samego zawodnika
                            if neighbor_data[2] == current_last_name and neighbor_data[3] == current_first_name:
                                reward += 4  # Nagroda za tego samego zawodnika

                            # Nagroda za konie o tym samym nazwisku
                            elif neighbor_data[2] == current_last_name:
                                reward += 1  # Nagroda za konie o tym samym nazwisku

                            # Nagroda za konie z tego samego zespołu
                            if neighbor_data[6] == current_team:
                                reward += 3  # Nagroda za konie z tego samego zespołu

                            # Nagroda za konie z tego samego kraju pochodzenia
                            if neighbor_data[4] == current_country:
                                reward += 2  # Nagroda za ten sam kraj pochodzenia

                            if (current_gender == "Stallion" and neighbor_data[5] == "Mare") or \
                                    (current_gender == "Mare" and neighbor_data[5] == "Stallion"):
                                reward -= 10  # Kara za umieszczenie ogiera obok klaczy


                            # Kara za sąsiedztwo ogiera i innego ogiera z innego zawodnika
                            if current_gender == "Stallion" and neighbor_data[5] == "Stallion":
                                # Sprawdzenie, czy ogiery mają różnych zawodników (imię i nazwisko)
                                if current_last_name != neighbor_data[2] or current_first_name != neighbor_data[3]:
                                    reward -= 5  # Kara za sąsiedztwo ogiera i innego ogiera
                        if (neighbor["type"] == "healing_box" or neighbor["type"] == "antidoping_box") and current_gender == "Gelding":
                            reward += 3
                # Kara za bezpośrednie sąsiedztwo ogiera z innym ogierem (różni zawodnicy)
                for nx, ny in direct_neighbors:
                    if (nx, ny) in self.grid_contents:
                        neighbor = self.grid_contents[(nx, ny)]
                        if neighbor["type"] == "horse":
                            neighbor_data = neighbor["data"]
                            # Kara tylko dla bezpośrednich pól
                            if current_gender == "Stallion" and neighbor_data[5] == "Stallion":
                                if current_last_name != neighbor_data[2] or current_first_name != neighbor_data[3]:
                                    reward -= 10  # Kara za bezpośrednie sąsiedztwo ogierów z różnych zawodników
                            if (current_gender == "Stallion" and neighbor_data[5] == "Gelding") or \
                                    (current_gender == "Gelding" and neighbor_data[5] == "Stallion"):
                                reward += 3  # Nagroda za ogiera obok wałacha





            else:
                reward -= 5  # Kara za próbę niewłaściwego umieszczenia

            target_x, target_y = self.find_closest_target(self.agent_position)

            # Przesunięcie agenta na wybrane pole
            self.agent_position = np.array([target_x, target_y], dtype=np.int32)

        # Umieszczenie boksu leczącego
        elif action == 5:
            if self.healing_boxes_remaining > 0 and (x, y) not in self.grid_contents:
                self.place_healing_box((x, y))
                self.encoded_grid_contents = encode_grid_contents(self.grid_contents,self.encoded_grid_contents, self.grid_size, self.max_horse_id, self.max_horse_team)

                #reward += 5  # Nagroda za umieszczenie boksu leczącego

                # Zamykamy pole
                self.stable_list[x][y] = 4
                # Nagrody podstawowe
                base_reward_6 = 10
                base_reward_7 = 10

                # Minimalne odległości do pól wartościowych
                min_distance_6 = self.calculate_min_distance((x, y), self.fields_6)
                min_distance_7 = self.calculate_min_distance((x, y), self.fields_7)

                # Dolicz nagrodę za bycie blisko pól
                if min_distance_6 > 0:
                    reward += (base_reward_6 / min_distance_6)  # Nagroda za bliskość do pól o wartości 6
                if min_distance_7 > 0:
                    reward += (base_reward_7 / min_distance_7)  # Nagroda za bliskość do pól o wartości 7

                # Nagroda za sąsiedztwo boksu specjalnego
                neighbors = self.get_neighbors((x, y))
                for nx, ny in neighbors:
                    if (nx, ny) in self.grid_contents:
                        neighbor = self.grid_contents[(nx, ny)]
                        if neighbor["type"] == "antidoping_box" or neighbor["type"] == "healing_box":
                            reward += 2
                        # Nagroda za sąsiedztwo z wałachem
                        elif neighbor["type"] == "horse" and neighbor["data"][5] == "Gelding":
                            reward += 1
            elif self.healing_boxes_remaining == 0:
                reward -= 10
            else:
                reward -= 5  # Kara za próbę umieszczenia boksu tam, gdzie nie można

            target_x, target_y = self.find_closest_target(self.agent_position)

            # Przesunięcie agenta na wybrane pole
            self.agent_position = np.array([target_x, target_y], dtype=np.int32)

        # Umieszczenie boksu antydopingowego
        elif action == 6:
            if self.antidoping_boxes_remaining > 0 and (x, y) not in self.grid_contents:
                self.place_antidoping_box((x, y))
                self.encoded_grid_contents = encode_grid_contents(self.grid_contents,self.encoded_grid_contents, self.grid_size, self.max_horse_id, self.max_horse_team)

                #reward += 5  # Nagroda za umieszczenie boksu antydopingowego

                # Zamykamy pole
                self.stable_list[x][y] = 4
                # Nagrody podstawowe
                base_reward_6 = 10
                base_reward_7 = 10

                # Minimalne odległości do pól wartościowych
                min_distance_6 = self.calculate_min_distance((x, y), self.fields_6)
                min_distance_7 = self.calculate_min_distance((x, y), self.fields_7)

                # Dolicz nagrodę za bycie blisko pól
                if min_distance_6 > 0:
                    reward += (base_reward_6 / min_distance_6)  # Nagroda za bliskość do pól o wartości 6
                if min_distance_7 > 0:
                    reward += (base_reward_7 / min_distance_7)  # Nagroda za bliskość do pól o wartości 7

                # Nagroda za sąsiedztwo boksu specjalnego
                neighbors = self.get_neighbors((x, y))
                for nx, ny in neighbors:
                    if (nx, ny) in self.grid_contents:
                        neighbor = self.grid_contents[(nx, ny)]
                        if neighbor["type"] == "antidoping_box" or neighbor["type"] == "healing_box":
                            reward += 2
                        # Nagroda za sąsiedztwo z wałachem
                        elif neighbor["type"] == "horse" and neighbor["data"][5] == "Gelding":
                            reward += 1
            elif self.antidoping_boxes_remaining == 0:
                reward -= 10
            else:
                reward -= 5  # Kara za próbę niewłaściwego umieszczenia

            target_x, target_y = self.find_closest_target(self.agent_position)

            # Przesunięcie agenta na wybrane pole
            self.agent_position = np.array([target_x, target_y], dtype=np.int32)

        # Stan zakończenia
        if self.horses_remaining == 0 and self.healing_boxes_remaining == 0 and self.antidoping_boxes_remaining == 0:
            done = True
            reward += 10  # Duża nagroda po ukończeniu zadania

        if not np.any(self.stable_list == 1) and not np.any(self.stable_list == 2):
            done = True  # Koniec, jeśli brak wolnych pól

        # Zwróć nowy stan, nagrodę, informację o zakończeniu i dodatkowe info




        observation = flatten(
            self.original_observation_space,
            {
                "stable": self.stable_list,
                "agent_position": self.agent_position,
                "horse_list": self.encoded_horse_list,
                "grid_contents": self.encoded_grid_contents,
                "current_horse_index": np.array([self.current_horse_index]),
            },
        )

        if self.normalize_output:
            # Dynamiczna aktualizacja statystyk
            self.update_normalization_stats(observation)
            self.update_reward_stats(reward)
            observation = self.normalize_observation(observation)
            reward = self.normalize_reward(reward)

        truncated = False
        return observation, reward, done, truncated, {}
