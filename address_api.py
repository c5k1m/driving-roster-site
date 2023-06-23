import requests
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Constants
COLLEGE_2_COORD_DICT = { # Convert College or Apartments to Lat, Long
    'Revelle': (32.87446225, -117.24098209937156),
    'Muir': (32.878014199999996, -117.24124330827681),
    'Marshall': (32.8817384, -117.24125091155366),
    'Warren': (32.88276945, -117.2340480725121),
    'ERC': (32.885091200000005, -117.24220110581237),
    'Sixth': (32.880474750000005, -117.24217019235094),
    'Seventh': (32.8881754, -117.2421452),
    'Pepper Canyon': (32.8792306, -117.2317966667491),
    'Rita Atkinson': (32.87257245, -117.23517887907337)
}

@app.put("/")
def func(input: dict[str, str] = None):
    # with ThreadPool() as p:
    #     results = p.map(geo_convert_multiprocess, input.items())
    results = []
    with ThreadPoolExecutor() as executor:
        future_to_address = {executor.submit(geo_convert_multiprocess, item): item for item in input.items()}
        for future in concurrent.futures.as_completed(future_to_address):
            address = future_to_address[future]
            try:
                data = future.result()
                results.append(data)
            except Exception as exc:
                print('%r generated an exception: %s' % (address, exc))
    return results

def geo_convert_multiprocess(input):
    name = input[0]
    address = input[1]
    try:
        if address in COLLEGE_2_COORD_DICT:
            # Return coord if it's from a known college
            return {name: COLLEGE_2_COORD_DICT[address]}
        base_url = f"https://nominatim.openstreetmap.org/search/{address}?format=json&addressdetails=1&limit=1&polygon_svg=1"
        r = requests.get(base_url).json()
        if len(r) == 0:
            return {name: (None, None)}
        lat = float(r[0]["lat"])
        long = float(r[0]["lon"])

        return {name : (lat, long)}
    except Exception as e:
        return {name: (None, None)}