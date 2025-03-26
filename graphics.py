import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# pip install moviepy      (si no lo tienes instalado)
from moviepy import ImageClip, TextClip, CompositeVideoClip, concatenate_videoclips


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
#draw_grafic_for_flags(df)

def video(labels, imagenes, video_name):
    clips = []

    for img, titulo in zip(imagenes, titulos):
        # Crea un clip a partir de la imagen. La duración se puede ajustar (ej. 3 segundos)
        imagen_clip = ImageClip(img).set_duration(3)
        
        # Crea un clip de texto para el título. Ajusta fuente, tamaño, color y posición según necesites.
        texto_clip = TextClip(titulo, fontsize=50, color='white', font='Amiri-Bold')
        texto_clip = texto_clip.set_duration(3).set_position(("center", "bottom"))
        
        # Superpone el texto sobre la imagen
        clip_composite = CompositeVideoClip([imagen_clip, texto_clip])
        clips.append(clip_composite)

    # Une todos los clips en un solo video
    video_final = concatenate_videoclips(clips, method="compose")

    # Guarda el video en un archivo
    video_final.write_videofile(f"{video_name}.mp4", fps=24)



# Lista de nombres de archivo de las imágenes
imagenes = ["/home/yesi/Documentos/FAMAF/ComputacionParalela/fire_spread_lab2025/1999_27j_S_burned_probabilities_O0.png",
            "/home/yesi/Documentos/FAMAF/ComputacionParalela/fire_spread_lab2025/1999_27j_S_burned_probabilities_O1.png",
            "/home/yesi/Documentos/FAMAF/ComputacionParalela/fire_spread_lab2025/1999_27j_S_burned_probabilities_O2.png" 
            "/home/yesi/Documentos/FAMAF/ComputacionParalela/fire_spread_lab2025/1999_27j_S_burned_probabilities_O3.png"]

# Lista de títulos para cada imagen (uno por ejecución, por ejemplo)
titulos = ["-O0", "-O1", "-O2","-O3"]

video(titulos, imagenes, "video_prueba")
