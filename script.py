import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re
import json

#FLAGS= "-O3 -march=native -ftree-vectorize -funroll-loops -ffast-math -fopt-info-vec-optimized"
FLAGS = "-Ofast -march=native -mfma -mavx2 -ftree-vectorize -funroll-loops -ffast-math -fopt-info-vec-optimized -fopenmp-simd"
CODE_FILE = "./graphics/burned_probabilities_data"

GCC_FLAGS_TO_TEST = {
    0: "-O0",
    1: "-O1",
    2: "-O2",
    3: "-O3",
    7: "-march=native -O1", # Optimización básica y para la arquitectura local
    8: "-march=native -O2",
    9: "-march=native -O3",
    10: "-O2 -march=native -flto",  # Optimización intermedia, arquitectura local y optimización en tiempo de enlace
    11: "-O3 -march=native -flto -funroll-loops",  # Máxima optimización, desenrollado de bucles
    12: "-ffast-math",
    13: "-march=native -ffast-math -O1",
    14: "-march=native -ffast-math -O2",
    15: "-march=native -ffast-math -O3",
}

GCC_FLAGS_VECT = {
    0: "-O1 -march=native -mfma -mavx2 -ftree-vectorize -funroll-loops -ffast-math -fopt-info-vec-optimized -fopenmp-simd",
    1: "-O2 -march=native -mfma -mavx2 -ftree-vectorize -funroll-loops -ffast-math -fopt-info-vec-optimized -fopenmp-simd",
    2: "-O3 -march=native -mfma -mavx2 -ftree-vectorize -funroll-loops -ffast-math -fopt-info-vec-optimized -fopenmp-simd",
    3: "-Ofast -march=native -mfma -mavx2 -ftree-vectorize -funroll-loops -ffast-math -fopt-info-vec-optimized -fopenmp-simd",
    4: "-O1 -march=native -ftree-vectorize -fopt-info-vec-optimized -fopenmp-simd",
    5: "-O2 -march=native -ftree-vectorize -fopt-info-vec-optimized -fopenmp-simd",
    6: "-O3 -march=native -ftree-vectorize -fopt-info-vec-optimized -fopenmp-simd",
    7: "-Ofast -march=native -ftree-vectorize -fopt-info-vec-optimized -fopenmp-simd",
    8: "-O1 -march=native -ftree-vectorize -fopt-info-vec-optimized -fopenmp-simd -funroll-loops",
    9: "-O2 -march=native -ftree-vectorize -fopt-info-vec-optimized -fopenmp-simd -funroll-loops",
    10: "-O3 -march=native -ftree-vectorize -fopt-info-vec-optimized -fopenmp-simd -funroll-loops",
    11: "-Ofast -march=native -ftree-vectorize -fopt-info-vec-optimized -fopenmp-simd -funroll-loops",
    12: "-O1 -march=native -ftree-vectorize -fopt-info-vec-optimized -fopenmp-simd -funroll-loops -flto",
    13: "-O2 -march=native -ftree-vectorize -fopt-info-vec-optimized -fopenmp-simd -funroll-loops -flto",
    14: "-O3 -march=native -ftree-vectorize -fopt-info-vec-optimized -fopenmp-simd -funroll-loops -flto",
    15: "-Ofast -march=native -ftree-vectorize -fopt-info-vec-optimized -fopenmp-simd -funroll-loops -flto",
    16: "-O0",

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
VALUES_TO_MAXIMIZE = ["insn_per_cycle", "cells_procesed_per_micro_sec"]


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

def run_gcc_with_all_flags(data:str, amount_of_tries=10, compiler:str="g++") -> list:
    stats = []
    for flag_id, flags in GCC_FLAGS_VECT.items():
        perf_stats = {}
        subprocess.run("make clean", shell=True)
        subprocess.run(f"make CXX={compiler} EXTRACXXFLAGS='{flags}'", shell=True)
        for n in range(amount_of_tries):
            result = subprocess.run(f"perf stat ./{CODE_FILE} {data}", shell=True, stderr=subprocess.PIPE, text=True)
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

def run_all_cases(amount_of_tries:int = 1) -> list:
    stats = []
    subprocess.run("make clean", shell=True)
    subprocess.run(f"make CXX='icpx' EXTRACXXFLAGS='{FLAGS}'" , shell=True)
    for data_id, data in DATA_TO_USE.items():
        print (f"Running data {data_id}, {data[0]}")
        perf_stats = {}
        if data[1] > 1000000:
            amount_of_tries = 5
        else:
            amount_of_tries = 30
        for n in range(amount_of_tries):
            print(f"Try number {n}")
            result = subprocess.run(f"perf stat ./{CODE_FILE} {data[0]}", shell=True, stderr=subprocess.PIPE, text=True)
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
        perf_stats.update({"flag": FLAGS, "data_name": data[0], "size_of_matrix": data[1]})
        print(perf_stats)
        stats.append(perf_stats.copy())
    return stats

def run_with_different_amount_of_simulations(data:str, amount_of_tries:int = 1) -> list:
    stats = []
    amount_of_sims = [128, 256, 512, 1024, 2048]
    for sim_amount in amount_of_sims:
        subprocess.run("make clean", shell=True)
        subprocess.run(f"make EXTRACXXFLAGS='{FLAGS}'  DEFINES=-DSIMULATIONS={sim_amount}", shell=True)
        for k in range(6):
            perf_stats = {}
            result = subprocess.run(f"perf stat ./{CODE_FILE} {data}", shell=True, stderr=subprocess.PIPE, text=True)
            last_value =  parse_perf_output(result.stderr)
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
            perf_stats.update({"data_name": data, "simulations": sim_amount})

            stats.append(perf_stats.copy())
    return stats


data_file = "./data/1999_27j_S"

stats = run_all_cases(30)
#stats = run_gcc_with_all_flags(data_file, 30)
df = pd.DataFrame(stats)

# Convertir la columna de flags a string para mejor visualización en los gráficos
df["flag"] = df["flag"].astype(str)

df.to_csv(f"csv_info/intrinsics.csv", index=False)


