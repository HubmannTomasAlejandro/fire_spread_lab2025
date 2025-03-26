import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re

GCC_FLAGS_TO_TEST = {
    0: "-O0",
    1: "-O1",
    2: "-O2",
    3: "-O3",
    #4: "-ffast-math",
    #6: "-march=native -O0",
    7: "-march=native -O1", # Optimización básica y para la arquitectura local
    8: "-march=native -O2",
    9: "-march=native -O3",
    #10: "-march=native -ffast-math -O1",
    #11: "-march=native -ffast-math -O2",
    #12: "-march=native -ffast-math -O3",
    10: "-O2 -march=native -flto",  # Optimización intermedia, arquitectura local y optimización en tiempo de enlace
    11: "-O3 -march=native -flto -funroll-loops",  # Máxima optimización, desenrollado de bucles
    12: "-ffast-math",
    13: "-march=native -ffast-math -O1",
    14: "-march=native -ffast-math -O2",
    15: "-march=native -ffast-math -O3",
}


DATA_TO_USE = {
    1: ("./data/1999_27j_S", 1157 * 1282),
    2: ("./data/1999_28", 334 * 282),
    3: ("./data/2000_8", 88 * 91),
    4: ("./data/2005_6", 119 * 119),
    5: ("./data/2005_26", 52 * 49),
    6: ("./data/2009_3", 117 * 136),
    7: ("./data/2011_19E", 218 * 223),
    8: ("./data/2011_19W", 125 * 105),
    9: ("./data/2015_50", 2917 * 3577), #1
    10: ("./data/2021_865", 1961 * 2395), #2
}

VALUES_TO_MINIMIZE = ["cycles", "branch_misses", "time_elapsed", "user_time", "sys_time"]
VALUES_TO_MAXIMIZE = ["insn_per_cycle"]


def parse_perf_output(perf_output: str) -> dict:
    """
    Parses the output from `perf stat` and extracts performance metrics into a dictionary.
    """
    perf_data = {}

    # Define regex patterns for extracting metrics
    patterns = {
        "cells_procesed_per_micro_sec": r"cells_burned_per_micro_sec:\s*([\d\.]+)",
        "cycles": r"([\d,]+) +cycles",
        "insn_per_cycle": r"([\d,.]+) +insn per cycle",
        "branches": r"([\d,]+) +branches",
        "branch_misses": r"([\d,]+) +branch-misses",
        "time_elapsed": r"([\d,.]+) seconds time elapsed",
        "user_time": r"([\d,.]+) seconds user",
        "sys_time": r"([\d,.]+) seconds sys",
        "instructions": r"([\d,]+) +instructions",
    }

    # Convert matches into dictionary entries
    for key, pattern in patterns.items():
        match = re.search(pattern, perf_output)
        if match:
            value = match.group(1).replace(",", "")  # Remove points from numbers
            perf_data[key] = float(value) if "." in value else int(value)

    return perf_data

def run_gcc_with_all_flags(code:str,  data:str, amount_of_tries=10, compiler:str="g++") -> list:
    stats = []
    for flag_id, flags in GCC_FLAGS_TO_TEST.items():
        perf_stats = {}
        subprocess.run("make clean", shell=True)
        subprocess.run(f"make CXX={compiler} EXTRACXXFLAGS='{flags}'", shell=True)
        for n in range(amount_of_tries):
            result = subprocess.run(f"perf stat ./{code} {data}", shell=True, stderr=subprocess.PIPE, text=True)
            last_value =  parse_perf_output(result.stderr)
            if not perf_stats:
                perf_stats = last_value.copy()
            else:
                for key, value in last_value.items():
                    if key in perf_stats:
                        if key in VALUES_TO_MINIMIZE:
                            perf_stats[key] = min(perf_stats[key], value)
                        elif key in VALUES_TO_MAXIMIZE:
                            perf_stats[key] = max(perf_stats[key], value)
                        else:
                            perf_stats[key] = value  # Add the current value to the sum
                    else:
                        perf_stats[key] = value  # If the key doesn't exist, just set it
            # Average the statistics by dividing the accumulated sum by the number of tries
        print(perf_stats)
        perf_stats.update({"flag": flags})
        stats.append(perf_stats.copy())
    return stats

def run_all_cases(code:str, amount_of_tries:int = 1) -> list:
    stats = []
    flags= "-O3"
    subprocess.run("make clean", shell=True)
    subprocess.run(f"make EXTRACXXFLAGS='{flags}'" , shell=True)
    for data_id, data in DATA_TO_USE.items():
        print (f"Running data {data_id}, {data[0]}")
        perf_stats = {}
        if data[1] > 1000000:
            amount_of_tries = 5
        else:
            amount_of_tries = 30
        for n in range(amount_of_tries):
            print(f"Try number {n}")
            result = subprocess.run(f"perf stat ./{code} {data[0]}", shell=True, stderr=subprocess.PIPE, text=True)
            last_value =  parse_perf_output(result.stderr)
            if not perf_stats:
                perf_stats = last_value.copy()
            for key, value in last_value.items():
                if key in perf_stats:
                    if key in VALUES_TO_MINIMIZE:
                        perf_stats[key] = min(perf_stats[key], value)
                    elif key in VALUES_TO_MAXIMIZE:
                        perf_stats[key] = max(perf_stats[key], value)
                    else:
                        perf_stats[key] = value  # Add the current value to the sum
                else:
                    perf_stats[key] = value  # If the key doesn't exist, just set it
        # Average the statistics by dividing the accumulated sum by the number of tries
        perf_stats.update({"flag": flags, "data_name": data[0], "size_of_matrix": data[1]})
        print(perf_stats)
        stats.append(perf_stats.copy())
    return stats

def run_with_different_amount_of_simulations(data:str, amount_of_tries:int = 1) -> list:
    code_file = "./graphics/burned_probabilities_data"
    stats = []
    flags= "-O3"
    amount_of_sims = [128, 256, 512, 1024, 2048]
    for sim_amount in amount_of_sims:
        subprocess.run("make clean", shell=True)
        subprocess.run(f"make EXTRACXXFLAGS='{flags}'  DEFINES=-DSIMULATIONS={sim_amount}", shell=True)
        for k in range(36):
            perf_stats = {}
            result = subprocess.run(f"perf stat ./{code_file} {data}", shell=True, stderr=subprocess.PIPE, text=True)
            last_value =  parse_perf_output(result.stderr)
            if not perf_stats:
                perf_stats = last_value.copy()
            for key, value in last_value.items():
                if key in perf_stats:
                    perf_stats[key] += value  # Add the current value to the sum
                else:
                    perf_stats[key] = value  # If the key doesn't exist, just set it
            # Average the statistics by dividing the accumulated sum by the number of tries
            #print(perf_stats)
            perf_stats.update({"data_name": data, "simulations": sim_amount})

            stats.append(perf_stats.copy())
    return stats


code_file = "./graphics/burned_probabilities_data"
data_file = "./data/2005_6"
"""
stats = run_all_cases(code_file, 1)
for i in range(len(stats)):
    time_elapsed = stats[i]['time_elapsed']
    instructions = stats[i]['instructions']
    data_name = stats[i]['data_name']
    size_of_matrix = stats[i]['size_of_matrix']
    print("\n****************************************************************************************")
    print(f"Data_field: {data_name}, size of matrix: {size_of_matrix}")
    print(f"Time elapsed: {time_elapsed} seconds")
    print("****************************************************************************************\n")
"""

"""
stats = run_gcc_with_all_flags(code_file, data_file,1)
# print(stats)
for i in range(len(stats)):
    time_elapsed = stats[i]['time_elapsed']
    instructions = stats[i]['insn_per_cycle']
    flag = stats[i]['flag']
    print("\n****************************************************************************************")
    print(f"Flag: {flag}")
    print(f"Time elapsed: {time_elapsed} seconds")
    print("****************************************************************************************\n")
"""
"""
stats = run_with_different_amount_of_simulations(data_file, 1)

with open("data.json", "w") as json_file:
    json.dump(stats, json_file, indent=4)
for i in range(len(stats)):
    time_elapsed = stats[i]['time_elapsed']
    cells_per_micro_sc = stats[i]['cells_procesed_per_micro_sec']
    flag = stats[i]['simulations']
    print("\n****************************************************************************************")
    print(f"Time elapsed: {time_elapsed} seconds with {flag} simulations")
    print(f"Cells processed per micro second: {cells_per_micro_sc}")
    print("****************************************************************************************\n")


"""

stats = run_all_cases(code_file, 1)
df = pd.DataFrame(stats)

# Convertir la columna de flags a string para mejor visualización en los gráficos
df["flag"] = df["flag"].astype(str)

df.to_csv(f"csv_info/prueba.csv", index=False)


