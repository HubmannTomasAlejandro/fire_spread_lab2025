import subprocess
import numpy as np
import matplotlib.pyplot as plt
import re

GCC_FLAGS_TO_TEST = {
    1: "-O1",
    2: "-O2",
    3: "-O3",
    4: "flto",
    5: "-march=native",
    6: "-fprofile-generate",
    7: "-fprofile-use",
    8: "-g",
    9: "-flop-block",
    10: "-funroll-loops",
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
        "task_clock": r"([\d,]+) msec task-clock",
        "context_switches": r"([\d,]+) +context-switches",
        "cpu_migrations": r"([\d,]+) +cpu-migrations",
        "page_faults": r"([\d,]+) +page-faults",
        "cycles": r"([\d,]+) +cycles",
        "stalled_cycles_frontend": r"([\d,]+) +stalled-cycles-frontend",
        "instructions": r"([\d,]+) +instructions",
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
            value = match.group(1).replace(",", "")  # Remove commas from numbers
            perf_data[key] = float(value) if "." in value else int(value)

    return perf_data

def run_gcc_with_all_flags(code:str,  data:str, amount_of_tries:int = 1) -> list:
    stats = []
    perf_stats = {}
    for flag_id, flags in GCC_FLAGS_TO_TEST.items():
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
        for key in perf_stats:
            perf_stats[key] /= amount_of_tries

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
        for key in perf_stats:
            perf_stats[key] /= amount_of_tries
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
for i in range(len(stats)):
    time_elapsed = stats[i]['time_elapsed']
    instructions = stats[i]['instructions']
    flag = stats[i]['flag']
    print("\n****************************************************************************************")
    print(f"Flag: {flag}")
    print(f"Time elapsed: {time_elapsed} seconds")
    print("****************************************************************************************\n")