import subprocess
import numpy as np
import matplotlib.pyplot as plt
import re

GCC_FLAGS_TO_TEST = {
    0: "-O0",
    1: "-O1",
    2: "-O2",
    3: "-O3",
    4: "-ffast-math",
    5: "-funroll-loops",
    6: "-march=native",
    7: "-march=native -O1",
    8: "-march=native -O2",
    9: "-march=native -O3",
    10: "-march=native -ffast-math -01",
    11: "-march=native -ffast-math -O2",
    12: "-march=native -ffast-math -O3",
}

DATA_TO_USE = {
    5: ("./data/2005_26", 52 * 49),
    1: ("./data/1999_27j_S", 1157 * 1282),
    2: ("./data/1999_28", 334 * 282),
    3: ("./data/2000_8", 88 * 91),
    4: ("./data/2005_6", 119 * 119),
    6: ("./data/2008", 88 * 91),
    7: ("./data/2009_3", 117 * 136),
    8: ("./data/2011_19E", 218 * 223),
    9: ("./data/2011_19W", 125 * 105),
    10: ("./data/2015_50", 2917 * 3577),
    11: ("./data/2021_865", 1961 * 2395),
}


def parse_perf_output(perf_output: str) -> dict:
    """
    Parses the output from `perf stat` and extracts performance metrics into a dictionary.
    """
    perf_data = {}

    # Define regex patterns for extracting metrics
    patterns = {
        "cycles": r"([\d,]+) +cycles",
        "insn_per_cycle": r"([\d,.]+) +insn per cycle",
        "branches": r"([\d,]+) +branches",
        "branch_misses": r"([\d,]+) +branch-misses",
        "time_elapsed": r"([\d,.]+) seconds time elapsed",
        "user_time": r"([\d,.]+) seconds user",
        "sys_time": r"([\d,.]+) seconds sys",
    }

    # Convert matches into dictionary entries
    for key, pattern in patterns.items():
        match = re.search(pattern, perf_output)
        if match:
            match = match.group(1).replace(".", "")  # Remove points from numbers
            value = match.replace(",", ".")  # Remove commas from numbers
            perf_data[key] = float(value) if "." in value else int(value)

    print(perf_data)
    return perf_data

def run_gcc_with_all_flags(code:str,  data:str, amount_of_tries:int = 2) -> list:
    stats = []
    for flag_id, flags in GCC_FLAGS_TO_TEST.items():
        perf_stats = {}
        subprocess.run("make clean", shell=True)
        subprocess.run(f"make EXTRACXXFLAGS={flags}", shell=True)
        for n in range(amount_of_tries):
            result = subprocess.run(f"perf stat ./{code} {data}", shell=True, stderr=subprocess.PIPE, text=True)
            last_value =  parse_perf_output(result.stderr)
            if not perf_stats:
                perf_stats = last_value.copy()
            else:
                for key, value in last_value.items():
                    if key in perf_stats:
                        perf_stats[key] += value  # Add the current value to the sum
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
    for data_id, data in DATA_TO_USE.items():
        subprocess.run("make clean", shell=True)
        subprocess.run(f"make EXTRACXXFLAGS={flags}", shell=True)
        perf_stats = {}
        for n in range(amount_of_tries):
            result = subprocess.run(f"perf stat ./{code} {data[0]}", shell=True, stderr=subprocess.PIPE, text=True)
            last_value =  parse_perf_output(result.stderr)
            if not perf_stats:
                perf_stats = last_value.copy()
            for key, value in last_value.items():
                if key in perf_stats:
                    perf_stats[key] += value  # Add the current value to the sum
                else:
                    perf_stats[key] = value  # If the key doesn't exist, just set it
        # Average the statistics by dividing the accumulated sum by the number of tries
        print(perf_stats)
        perf_stats.update({"flag": flags, "data_name": data[0], "size_of_matrix": data[1]})
        stats.append(perf_stats.copy())
    return stats

code_file = "./graphics/burned_probabilities_data"
data_file = "./data/1999_27j_S"

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

stats = run_gcc_with_all_flags(code_file, data_file, 1)
print(stats)
for i in range(len(stats)):
    time_elapsed = stats[i]['time_elapsed']
    instructions = stats[i]['insn_per_cycle']
    flag = stats[i]['flag']
    print("\n****************************************************************************************")
    print(f"Flag: {flag}")
    print(f"Time elapsed: {time_elapsed} seconds")
    print("****************************************************************************************\n")
