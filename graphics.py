import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# pip install moviepy      (si no lo tienes instalado)
from moviepy import ImageClip, CompositeVideoClip, concatenate_videoclips
from moviepy.video.VideoClip import TextClip


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

#draw_grafic_for_flags(df)
#draw_generic_grafic(df,"size_of_matrix")

def video(labels, imagenes, video_name):
    clips = []

    for img, titulo in zip(imagenes, labels):
        # Crea un clip a partir de la imagen con duración de 3 segundos
        imagen_clip = ImageClip(img).with_duration(1)

        # Crea un clip de texto para el título con duración de 3 segundos y posición en el centro inferior.
        # Se especifica method="caption" para evitar el conflicto con el parámetro font.
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
        texto_clip = TextClip(font=font_path,text=titulo, font_size=25)
        texto_clip = texto_clip.with_position(("center", "top")).with_duration(1)

        # Superpone el texto sobre la imagen
        clip_composite = CompositeVideoClip([imagen_clip, texto_clip])
        clips.append(clip_composite)

    # Une todos los clips en un solo video
    video_final = concatenate_videoclips(clips, method="compose")

    # Guarda el video en un archivo MP4
    video_final.write_videofile(f"{video_name}.mp4", fps=24)



# Lista de nombres de archivo de las imágenes
imagenes = ["./Graficos Lab1/1999_27j_S_burned_probabilities_O0.png",
            "./Graficos Lab1/1999_27j_S_burned_probabilities_O1.png",
            "./Graficos Lab1/1999_27j_S_burned_probabilities_O2.png",
            "./Graficos Lab1/1999_27j_S_burned_probabilities_O3.png",
            "./Graficos Lab1/1999_27j_S_burned_probabilities_4.png",
            "./Graficos Lab1/1999_27j_S_burned_probabilities_5.png",
            "./Graficos Lab1/1999_27j_S_burned_probabilities_6.png",
            "./Graficos Lab1/1999_27j_S_burned_probabilities_7.png",
            "./Graficos Lab1/1999_27j_S_burned_probabilities_8.png",
            "./Graficos Lab1/1999_27j_S_burned_probabilities_9.png",
            "./Graficos Lab1/1999_27j_S_burned_probabilities_10.png",
            "./Graficos Lab1/1999_27j_S_burned_probabilities_11.png",
            "./Graficos Lab1/1999_27j_S_burned_probabilities_12.png"]

# Lista de títulos para cada imagen (uno por ejecución, por ejemplo)
titulos = ["-O0", "-O1", "-O2", "-O3","-march=native -O1","-march=native -O2","-march=native -O3",
           "-O2 -march=native -flto","-O3 -march=native -flto -funroll-loops","-ffast-math",
           "-march=native -ffast-math -O1","-march=native -ffast-math -O2",
           "-march=native -ffast-math -O3"]

#video(titulos, imagenes, "video_flags")

def float_double (df_to_use, df_type, palette, title):

    if not title:
        title = "Comparación de datos de punto flotante y doble"
    # Cargar los archivos CSV
    for i in range (len(df_to_use)):
        df = pd.read_csv(df_to_use[i])
        # Agregar una columna para identificar el tipo de dato
        df["Tipo"] = df_type[i]
        # Combinar los DataFrames
        if i == 0:
            df_final = df
        else:
            df_final = pd.concat([df_final, df])
    # Graficar usando Seaborn
    plt.figure(figsize=(18, 6))
    sns.barplot(data=df_final, x="data_name", y="cells_procesed_per_micro_sec", hue="Tipo", palette=palette)
    # Personalización del gráfico
    plt.xlabel("Data Name")
    plt.ylabel("cells_procesed_per_micro_sec (µs/cell)")
    plt.title(title)
    plt.legend(title="Tipo de dato")
    plt.xticks(rotation=0)
    plt.tight_layout()  # Asegurarse de que no se recorten las etiquetas

    # Mostrar el gráfico
    plt.show()


df_to_use = [
            "./csv_info/original_performance.csv",
            "./csv_info/run_all_cases_opt_post_presentacion.csv",
            "./csv_info/intrinsics.csv",
            ]

df_type = [
    "sin optimización",
    "Lab 1",
    "Lab 2"
    ]

#palette = ["#5caee3", "#ffb347", "#f25c54"]
#float_double(df_to_use=df_to_use, df_type=df_type, palette=palette ,title="Evolucion de rendimiento")



def draw_threads_grafic(df:pd.DataFrame, best_metric:float):
    cell_metric = df["cells_procesed_per_micro_sec"]
    threads = df["threads"]

    efficiency = [ s/p for s,p in zip(cell_metric, threads) ]

    # Threads vs cells_procesed_per_micro_sec
    plt.figure(figsize=(12, 6))
    plt.plot(threads, cell_metric, marker='o', color='darkviolet')
    plt.plot(threads, [best_metric]*len(threads), linestyle='--', color='red', label='Best result')
    plt.xlabel('Threads')
    plt.ylabel('cells_procesed_per_micro_sec (µs/cell)')
    plt.title('cells_procesed_per_micro_sec vs threads')
    plt.legend()
    plt.grid(True)
    plt.savefig('threads_2011_cell.png', dpi=300)
    plt.show()

    # Threads vs Eficiency
    plt.figure(figsize=(12, 6))
    plt.plot(threads, efficiency, marker='o', color='limegreen')
    plt.xlabel('Threads')
    plt.ylabel('Eficiency')
    plt.title('Eficiency vs threads')
    plt.grid(True)
    plt.savefig('threads_2011_ef.png', dpi=300)
    plt.show()


df1 = pd.read_csv("./csv_info/try_threads_omp2.csv")
draw_threads_grafic(df1, best_metric=1.447868)