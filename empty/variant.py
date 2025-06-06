import random
import os
from copy import deepcopy

N_VARIANTS = 10  # nombre de variantes à générer
WALL_PROB = 0.25  # probabilité de placer un mur vertical "|"
MAP_FILE = "5x5_empty.map"
OUTPUT_PREFIX = "5x5_var"

# Positions des lettres à placer
LETTERS = ["R", "G", "B", "Y"]

def read_map(filepath):
    with open(filepath, "r") as f:
        return [list(line.rstrip("\n")) for line in f]

def write_map(map_data, filepath):
    with open(filepath, "w") as f:
        for line in map_data:
            f.write("".join(line) + "\n")

def is_case(row, col, map_data):
    return map_data[row][col] == " "

def get_case_coords(map_data):
    coords = []
    for row in range(1, len(map_data)-1):
        for col in range(1, len(map_data[0])-1, 2):
            if is_case(row, col, map_data):
                coords.append((row, col))
    return coords

def generate_variant(base_map):
    map_data = deepcopy(base_map)

    # Ajout de murs aléatoires (uniquement "|" entre deux cases)
    for row in range(1, len(map_data)-1):
        for col in range(2, len(map_data[0])-2, 2):  # uniquement les colonnes qui peuvent avoir un "|"
            if map_data[row][col] == ":" and random.random() < WALL_PROB:
                map_data[row][col] = "|"

    # Placement des lettres RGBY dans des cases libres
    available_coords = get_case_coords(map_data)
    random.shuffle(available_coords)
    for letter in LETTERS:
        if not available_coords:
            break
        r, c = available_coords.pop()
        map_data[r][c] = letter

    return map_data

def main():
    base_map = read_map(MAP_FILE)
    for i in range(N_VARIANTS):
        variant = generate_variant(base_map)
        output_path = f"{OUTPUT_PREFIX}{i}.map"
        write_map(variant, output_path)
        print(f"✅ Variante sauvegardée : {output_path}")

if __name__ == "__main__":
    main()
