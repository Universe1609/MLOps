import pickle

import click
import numpy as np
import pandas as pd

np.seterr(all="ignore")


@click.command()
@click.option(
    "--input_data",
    "-i",
    help="Values of age, capital-gain, capital-loss and hours-per-week separated by comma",
)
def predict_adult_census(input_data: str) -> np.ndarray:
    """
    Runs a predition on the adult census data using a serialized logistic regression model.

    :param input_data: A string containing values for agem capital-gain, capital-loss, and hours-per-week
        separated by commas.
    :type input_data: str
    :return: An array of predicted values.
    :rtype: numpy.ndarray
    """

    # cargando el modelo serializado
    with open("../models/adult_census_lr.pkl", "rb") as f:
        trained_model = pickle.load(f)

    # Preparando la data como un DataFrame
    input_df = pd.DataFrame(
        [input_data.split(",")],
        columns=["age", "capital-gain", "capital-loss", "hours-per-week"],
    )
    input_df = input_df.astype(int)

    # HACIOENDO PREDICCIONES
    predictions = trained_model.predict(input_df)

    for prediction in predictions:
        click.echo(prediction)

    return predictions


if __name__ == "__main__":
    predict_adult_census()
