import sys
import numpy as np
import matplotlib.pyplot as plt
import os

GCC_FLAGS_TO_TEST = {
    1: "-01",
    2: "-02",
    3: "-03",
    4: "flto",
    5: "-march=native",
    6: "-fprofile-generate",
    7: "-fprofile-use",
    8: "-g",
    9: "-flop-block",
    10: "-funroll-loops",
    11: "-floop-info-vec-missed",
}

def run_gcc_with_all_flags(code:str,  data:str):
    for flag_id, flag in GCC_FLAGS_TO_TEST.items():
        os.system("make clean")
        os.system(f"make EXTRACXXFLAGS={flag}")
        result = os.popen(f"perf stat ./{code} {data}")
        print(result)

code_file = "./graphics/burned_probabilities_data"
data_file = "./data/1999_27j_S"
run_gcc_with_all_flags(code_file, data_file)