import pickle
import json
import numpy as np
__model = None
__locations = None
__data_columns = None

def get_estimated_price(location, noOfRooms):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = noOfRooms
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0])

def get_location_names():
    return __locations

def load_saved_artifacts():
    print('Loading saved Artifacts...start')
    global __data_columns
    global __locations

    with open("./artifacts/columns.json") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[1:]
    global __model
    with open("./artifacts/homegram_home_prices_model.pickle", 'rb') as f:
        __model = pickle.load(f)
    print("loading saved artifacts...done")

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price("bauchi ring road, jos", 3))