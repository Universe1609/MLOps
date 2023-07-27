# Importamos el módulo pickle para serializar y deserializar objetos de Python.
import pickle

# Importamos click para crear interfaces de línea de comandos en Python.
import click
# Importamos numpy para realizar operaciones numéricas y manipular matrices.
import numpy as np
# Importamos pandas para manipular y analizar datos.
import pandas as pd

# Configuramos numpy para ignorar todos los errores.
np.seterr(all="ignore")

# Definimos un nuevo comando de línea de comandos con click.
@click.command()
# Añadimos una opción al comando para introducir los datos de entrada.
@click.option(
    "--input_data",
    "-i",
    help="Valores de edad, ganancia de capital, pérdida de capital y horas por semana separados por comas",
)
def predict_adult_census(input_data: str) -> np.ndarray:
    """
    Función principal que usa un modelo de regresión logística previamente entrenado para hacer predicciones en base a los datos del censo de adultos.
    """

    # Cargamos el modelo serializado desde un archivo.
    with open("../models/adult_census_lr.pkl", "rb") as f:
        trained_model = pickle.load(f)

    # Creamos un DataFrame de pandas a partir de los datos de entrada.
    input_df = pd.DataFrame(
        [input_data.split(",")],
        columns=["age", "capital-gain", "capital-loss", "hours-per-week"],
    )
    # Cambiamos el tipo de datos del DataFrame a int.
    input_df = input_df.astype(int)

    # Utilizamos el modelo para hacer predicciones en base a los datos de entrada.
    predictions = trained_model.predict(input_df)

    # Imprimimos cada predicción en la consola.
    for prediction in predictions:
        click.echo(prediction)

    # Devolvemos las predicciones como resultado de la función.
    return predictions

# Si el script se está ejecutando como programa principal, llamamos a la función predict_adult_census.
if __name__ == "__main__":
    predict_adult_census()
