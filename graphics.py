import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


metrics = ["cells_procesed_per_micro_sec",'instructions','branches','time_elapsed',"cycles","insn_per_cycle","branch_misses", "user_time","sys_time",]
units = {
        'instructions': '',
        'branches': '',
        'time_elapsed': 's',
        "cells_procesed_per_micro_sec": 'µs/cell',
        "cycles": '',
        "insn_per_cycle": 'insn/cycle',
        "branch_misses": '',
        "user_time": 's',
        "sys_time": 's',
    }
# Identificar las demás métricas a graficar y su unidad de medición
def draw_grafic_for_flags(df:pd.DataFrame):
    draw_generic_grafic(df,"flag")

def draw_generic_grafic(df:pd.DataFrame, x_label:str):
    for metric in metrics:
        plt.figure(figsize=(18, 6))
        ax = sns.barplot(x=x_label, y=metric, data=df,color='mediumturquoise')

        # Agregar anotaciones en cada barra con formato en notación científica
        for container in ax.containers:
            # Crear etiquetas formateadas con notación científica y la unidad correspondiente
            if metric == "cells_procesed_per_micro_sec":
                labels = [f"{bar.get_height():.2f} {units.get(metric, '')}" for bar in container]
            else:
                labels = [f"{bar.get_height():.2e} {units.get(metric, '')}" for bar in container]
            ax.bar_label(container, labels=labels, label_type='edge', padding=3)

        plt.title(f"Comparación de {metric} por Flag")
        plt.xlabel("Flag de Optimización")
        # Agregar la unidad en la etiqueta del eje y
        plt.ylabel(f"{metric} ({units.get(metric, '')})")
        # Rotar las etiquetas del eje x para mayor legibilidad
        plt.xticks(rotation=45)
        plt.tight_layout()  # Asegurarse de que no se recorten las etiquetas
        #plt.savefig(f'grafico_{metric}.png', dpi=300)
        plt.show()

file_to_use = "csv_info/data_all_flags_2005.csv"

df = pd.read_csv(file_to_use)
draw_grafic_for_flags(df)