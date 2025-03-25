import json
import matplotlib.pyplot as plt
import seaborn as sns  
import pandas as pd 

# Cargar los datos desde el archivo JSON
with open("/home/yesi/Documentos/FAMAF/ComputacionParalela/fire_spread_lab2025/data_2005_26.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Crear el boxplot
plt.figure(figsize=(18, 8))
sns.boxplot(x="simulations", 
            y="cells_procesed_per_micro_sec", 
            data=df,
            flierprops = {'markerfacecolor':'deeppink'},
            color="mediumturquoise")
plt.xlabel("Cantidad de simulaciones")
plt.ylabel("Cells processed per micro sec")
plt.title("Boxplot de Cells processed per micro sec por simulación")
#plt.savefig('box_plot.png', dpi=300)
plt.show()
