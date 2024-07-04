#%%

import random
import pickle
import os
import pandas as pd

def generate_dummy_data(num_people: int = 1000, square_meters_per_animal: float = 3.2):
    animals = []
    areas = []
    for person_id in range(1, num_people + 1):
        animal = random.randint(0,5)
        animals.append(animal)
        areas.append(animal*square_meters_per_animal+random.uniform(0,1)*10)
    df = pd.DataFrame(data = list(zip(areas,animals)), columns = ["area", "animals"])
    return df

square_meters_per_animal = 3.2
num_people = 1000
df = generate_dummy_data(num_people = num_people, square_meters_per_animal = square_meters_per_animal)

target = []
for row in range(len(df)):
    free_area = df["area"].iloc[row]-square_meters_per_animal*df["animals"].iloc[row]
    if free_area > square_meters_per_animal:
        target.append(True)
    else:
        target.append(False)

df["target"] = target

# Specify the path to your pickle file
pickle_file_path = r'..\data\mock_data.pkl'

with open(pickle_file_path, 'wb') as pkl_file:
    pickle.dump(df, pkl_file)

print(f"Pickle file saved successfully at {pickle_file_path}")

# %%
