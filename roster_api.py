import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class InputParametersModel(BaseModel):
    driver_coord: list[list[int]]
    passenger_coord: list[list[int]]
    driver_names: list[str]
    passenger_names: list[str]
    strict_dict: dict[str, list[str]]
    flex_dict: dict[str, list[str]]

@app.put("/")
def root(input_parameters: InputParametersModel):
    # Process Input Data
    input_dict = {}
    for item in input_parameters:
        key = item[0]
        value = item[1]
        input_dict[key] = value
    
    # Create variables
    driver_coord = np.array(input_dict["driver_coord"])
    passenger_coord = np.array(input_dict["passenger_coord"])
    driver_names = np.array(input_dict["driver_names"])
    passenger_names = np.array(input_dict["passenger_names"])
    strict_dict = input_dict["strict_dict"]
    flex_dict = input_dict["flex_dict"]

    # Run Algorithm
    allocations = run_roster_algorithm(
        driver_coord,
        passenger_coord,
        driver_names,
        passenger_names,
        strict_dict,
        flex_dict
    )

    return allocations

def run_roster_algorithm(
    driver_coord: np.ndarray,
    passenger_coord: np.ndarray,
    driver_names: np.ndarray,
    passenger_names: np.ndarray,
    strict_dict: dict[str, list[str]],
    flex_dict: dict[str, list[str]]
) -> dict[str, list[str]]:
    return {"Hello": ["World"]}