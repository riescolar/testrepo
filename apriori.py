#!/usr/bin/env python
# coding: utf-8

# <img src="master.png">
# <!-- Técnicas Actuales de Análisis de Datos (I): Minería de Datos
# (c)2023 - Ángel F. Zazo Rodríguez
# Máster en Business Analytic & Data Science
# Instituto Universitario de Estudios de la Ciencia y la Tecnología
# Universidad de Salamanca -->
#
# #  Reglas de asociación. Algoritmo Apriori
#
# En este cuaderno vemos la aplicación del algoritmo Apriori sobre la colección
# Market_Basket_Optimisation que contiene 7501 transacciones que fueron
# registras por una tienda al sur de Francia de las compras que hicieron sus
# clientes en una semana.
#
# ----
# **Importante:** para poder utilizar el algoritmo Apriori es necesario que se
# instale previamente el módulo `apyori`. Para ello ejecute la siguiente orden:
#
#    * `pip install apyori` (puede ser necesario utilizar `pip3` en vez de `pip`)
#
# Nota: si utiliza la distribución **Anaconda3**, lance previamente una
# consola de *conda* (botón `Inicio` de Windows > `Anaconda3` >  `Anaconda
# Prompt`). Se incluye una captura de pantalla:
#
# <img style="width: 70%" src="conda_install_apyori.png">

# In[1]: Importación de la función
from apyori import apriori

# In[2]: # ### Carga de librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# %matplotlib inline


# In[3]: # ###  Obtención de datos
# Es recomendable ver antes el contenido del fichero, y luego buscar la ayuda
# de la función `pandas.read_csv` para determinar los mejores parámetros que
# se deben utilizar para su lectura. En este ejemplo solo se debe indicar que
# los datos no contienen cabecera.
fichero = "Market_Basket_Optimisation.csv"
dataset = pd.read_csv(fichero, header = None)


# In[4]: # ### Análisis exploratorio

# Visualización de las primeras observaciones
dataset.head()

# Podemos ver que cada observación es en realidad la lista de productos que se
# han comprado conjuntamente. El número máximo de productos adquiridos en la
# misma compra ha sido de 19 (el número de columnas), el número mínimo es 1.
 # Existe una gran cantidad de `NaN`, que introduce Pandas automáticamente.

# In[5]: # #### Procesado de datos
#
# * Obtengo lista de listas con cada transacción.
# * Sobre la marcha cambio los espacios por `" "` por `"_"` para no tener espacios en blanco que puedan confundir el procesamiento.
# * Sobre la marcha quito elementos duplicados que hay en 5 transacciones (utilizo `set`).
transactions = []
for i in range(len(dataset)):
    products = list()
    for x in dataset.values[i]:
        if str(x) != 'nan':
            products.append(x.replace(" ", "_"))
    transactions.append(list(set(products)))


# In[6]: # Vemos las 5 primeras transacciones
transactions[:5]

# In[7]: # #### Productos más frecuentes de transacciones de 1 elemento
from collections import Counter
uniques = [x[0] for x in transactions if len(x) == 1]
uniques = Counter(uniques).most_common()
x = [a for a,b in uniques]
y = [b for a,b in uniques]

## Los 40 productos más frecuentes
n = 40
plt.figure(figsize=(10,4))
ax = sns.barplot(x=x[:n], y=y[:n], color='steelblue')
ax.set_xticklabels(x[:n], rotation=90)
plt.show()

# In[8]: # #### Productos más frecuentes en total
uniques = [prod for products in transactions for prod in products]
uniques = Counter(uniques).most_common()
x = [a for a,b in uniques]
y = [b for a,b in uniques]

## Los 40 productos más frecuentes
n = 40
plt.figure(figsize=(10,4))
ax = sns.barplot(x=x[:n], y=y[:n], color='steelblue')
ax.set_xticklabels(x[:n], rotation=90)
plt.show()

# In[9]: # ## Algoritmo Apriori
rules = apriori(transactions, min_support = 0.006,
                min_confidence = 0.2, min_lift = 3)
rules = list(rules)


# In[10]: # #### Presentación de las reglas
df = pd.DataFrame(columns=["Rule", "Support", "Confidence", "Lift"])
for result in rules:
    rule = result[0]
    support = result[1]
    ordered = result[2][0]
    antecedente, consecuente, confidence, lift = ordered
    rule = " + ".join(antecedente) + ' -> ' + ' + '.join(consecuente)
#    print("Rule: %s" % rule)
#    print("Support: %.4f" % support)
#    print("Confidence: %.4f" % confidence)
#    print("Lift: %.4f" % lift)
#    print("=====================================")
    df.loc[len(df)] = [rule, support, confidence, lift]
df
