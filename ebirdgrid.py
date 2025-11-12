"""
Script to create a species per grid json file using data from ebird
Output file is an array of grid data ordered by longitude
"""

import json
import csv
from pathlib import Path
import argparse
import sys
import logging

csv.field_size_limit(sys.maxsize)
import math
import numpy as np
from dateutil.parser import parse as parse_date
from datetime import datetime
import matplotlib.pyplot as plt
import geopandas as gpd

NZ_RECT = [166.419, -34.0, 178.517093541, -47.29]

# maximum width height from kml file
MAX_LNG = 0.15556536514500863
MAX_LAT = 0.10025559492370206


def read_ebird_atlas_squares():
    
    global MAX_LNG, MAX_LAT

    kml_file = "Atlas Grid Squares with names_June 2020.kml"
    logging.info("Reading NZ ebird squares from %s",kml_file)
    gdf = gpd.read_file(kml_file, driver="KML")
    all_bounds = []
    max_lng = None
    max_lat = None
    for geo in gdf["geometry"]:
        all_bounds.append(geo.bounds)
        lng = geo.bounds[2] - geo.bounds[0]
        lat = geo.bounds[3] - geo.bounds[1]
        if max_lng is None or lng > max_lng:
            max_lng = lng
        if max_lat is None or lat > max_lat:
            max_lat = lat
    MAX_LNG = max_lng
    MAX_LAT = max_lat
    return all_bounds


def common_to_ebird():
    # generate dictionary from common name to ebird id name
    ebird_labels = {}
    with open("eBird_taxonomy_v2024.csv") as f:
        f.readline()

        for line in f:
            split_l = line.split(",")
            ebird_labels[split_l[4].lower()] = split_l[2]

    return ebird_labels


def find_region_meta(region_meta, lng, lat):
    for code, species_info in region_meta.items():
        region_bounds = species_info["region"]["info"]["bounds"]
        if (
            lng >= region_bounds["minX"]
            and lng <= region_bounds["maxX"]
            and lat >= region_bounds["minY"]
            and lat <= region_bounds["maxY"]
        ):
            species_list = species_info["species"]
            return species_list, species_info["region"]
    return None, None


def slow_find(squares, lng, lat):
    for i, square in enumerate(squares):
        bounds = square["bounds"]
        if (
            lng >= bounds[0]
            and lng <= bounds[2]
            and lat >= bounds[1]
            and lat <= bounds[3]
        ):
            # print("Found square for ", lng, lat, " as ",square)
            return i, square
    return None, None


def binary_grid_search(squares, lng, lat):
    high = len(squares)
    low = 0
    found = None
    # squares in order of lng so can binary search
    while high > low:
        mid = (high + low) // 2
        square = squares[mid]
        bounds = square["bounds"]

        if bounds[0] <= lng and bounds[2] >= lng:
            found = mid
            break
        if bounds[2] < lng:
            low = mid + 1
        else:
            high = mid - 1
    if found is None:
        logging.error("Could not find species square for %s, %s", lng, lat)
        return None

    mid = found

    # check forwards until out of lng or found match
    while mid < len(squares):
        square = squares[mid]
        bounds = square["bounds"]
        if bounds[0] > lng:
            break
        if bounds[1] <= lat and bounds[3] >= lat:
            return mid, square
        mid += 1

    mid = found - 1
    # check back until out of lng or found match
    while mid >= 0:
        square = squares[mid]
        bounds = square["bounds"]
        if bounds[0] > lng:
            break
        if bounds[1] <= lat and bounds[3] >= lat:
            return mid, square
        mid -= 1
    logging.error("Could not find species square for %s, %s", lng, lat)
    return None


def set_neighbours(squares):
    logging.info("Setting neighbours metadata of squares")
    max_lng = 0.16
    max_lat = 0.11

    max_width = None
    max_height = None
    for square in squares:
        neighbours = []
        bounds = square["bounds"]
        centre = [(bounds[2] + bounds[0]) / 2, ((bounds[1] + bounds[3]) / 2)]
        for index, other in enumerate(squares):
            if other == square:
                continue

            other_bounds = other["bounds"]
            other_centre = [
                (other_bounds[2] + other_bounds[0]) / 2,
                ((other_bounds[1] + other_bounds[3]) / 2),
            ]
            dist_lng = abs(other_centre[0] - centre[0])
            dist_lat = abs(other_centre[1] - centre[1])
            if dist_lng < max_lng and dist_lat < max_lat:
                neighbours.append(index)
        square["neighbours_i"] = neighbours


model_ebirds = {'gobwhi1', 'x01059', 'sobkiw2', 'kea1', 'whiteh1', 'blknod', 'sobkiw1', 'yellow2', 'rettro', 'criros2', 'kelgul', 'okbkiw1', 'shbcuc1', 'nezrob3', 'eursta', 'whiter', 'purswa6', 'yefpar3', 'comcha', 'auspip3', 'houspa', 'x00458', 'ausmag2', 'cangoo', 'oyster1', 'riflem1', 'comred', 'noiger1', 'mallar3', 'gryger1', 'parshe1', 'nezbel1', 'yellow3', 'nezrob2', 'sobkiw3', 'pacrob2', 'skylar', 'sbweye1', 'liskiw1', 'grskiw1', 'y01193', 'commyn', 'kiwi1', 'fernbi1', 'blkswa', 'nezpig2', 'lotkoe1', 'redpol1', 'parake', 'silver3', 'eurbla', 'nezfal1', 'sooter1', 'maslap1', 'tomtit1', 'eurgre1', 'dunnoc1', 'nezfan1', 'brncre', 'nezkak1', 'sackin3', 'baicra1', 'sonthr1', 'calqua', 'weka1', 'tui1', 'morepo2', 'nibkiw1'}
def labels_with_one_samples(grid_file):
    common_ebird_map = common_to_ebird()
    id_common ={}
    for k, v in common_ebird_map.items():
        id_common[v] = k
    with open(grid_file,"r") as f:
        metadata = json.load(f)
    low_data_birds =set()
    for square in metadata["grid_meta"]:
        species_meta = merge_neighbours(square, metadata["grid_meta"])
        found = False
        total_birds = 0
        for k,v in species_meta.items():
            total_birds += np.sum(list(v.values()))
            if k not in model_ebirds:
                continue
            if np.sum(list(v.values()))<=1:
                print(f"{id_common[k]}:{v}")
                low_data_birds.add(id_common[k])
                found = True
        if found:
            print( f"https://maps.google.com/?q={square["bounds"][1]},{square["bounds"][0]}")
            print(square["bounds"], " total birds ", total_birds)

    print("Low ids are ",low_data_birds)
def merge_neighbours(square, species_meta):
    species_per_month = square["species_per_month"].copy()
    for neighbour in square["neighbours_i"]:
        neighbour_species = species_meta[neighbour]["species_per_month"]
        for species, month_data in neighbour_species.items():
            if species not in species_per_month:
                species_per_month[species] = month_data.copy()
                continue
            for (
                m,
                c,
            ) in month_data.items():
                species_per_month[species][m] += c
    return species_per_month

def birds_at_location(grid_file,lat,lng):
        with open(grid_file,"r") as f:
            metadata = json.load(f)
        grid_data = metadata["grid_meta"]

def main():
    fmt = "%(process)d %(thread)s:%(levelname)7s %(message)s"
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S"
    )
    args = parse_args()

    squares = read_ebird_atlas_squares()
    squares = sorted(squares, key=lambda square: square[0])

    region_metafile = Path("ebird_species.json")
    logging.info("Reading regoin ebird metadata from %s",region_metafile)
    with region_metafile.open("r") as f:
        region_meta = json.load(f)
    grid_meta = []

    # match up region meta where applicable,
    # could cap species to what is in here to avoid one of or historical occurences
    for square in squares:
        lng = (square[2] + square[0]) / 2
        lat = (square[1] + square[3]) / 2

        _, region_bound = find_region_meta(region_meta, lng, lat)
        region_code = None
        if region_bound is not None:
            region_code = region_bound["code"]

        meta = {"region_code": region_code, "bounds": square, "species_per_month": {}}

        grid_meta.append(meta)

    common_ebird_map = common_to_ebird()

    logging.info("Loading grid data from ", args.csv)
    count = 0
    with args.csv.open("r") as f:
        dreader = csv.reader(f, delimiter="\t", quotechar="|")
        first = True
        headers = None
        name_i = None
        lat_i = None
        lng_i = None
        type_i = None
        date_i = None
        latest_date = None
        for row in dreader:
            count += 1
            if first:
                headers = row
                print(headers)
                name_i = headers.index("COMMON NAME")
                lat_i = headers.index("LATITUDE")
                lng_i = headers.index("LONGITUDE")
                date_i = headers.index("OBSERVATION DATE")
                type_i = headers.index("OBSERVATION TYPE")
                first = False
                continue
            lat = float(row[lat_i])
            lng = float(row[lng_i])
            res = binary_grid_search(grid_meta, lng, lat)
            if res is None:
                # shall we just add these, probably could not add them also generally in the middle of the oceaon
                grid_meta, square = add_new_square(grid_meta, lng, lat, region_meta)
            else:
                _, square = res

            common_name = row[name_i]
            ebird_id = common_ebird_map.get(common_name.lower(), "unknown")
            if ebird_id == "unknown":
                logging.warning("Unmatched bird %s %s", common_name, row)
                continue
            obs_date = parse_date(row[date_i])
            if latest_date is None or obs_date > latest_date:
                latest_date = obs_date
            species_dic = square["species_per_month"]
            if ebird_id not in species_dic:

                month_count = {}
                for i in range(12):
                    month_count[i + 1] = 0
                species_dic[ebird_id] = month_count
            species_dic[ebird_id][obs_date.month] += 1

            if count % 1000 == 0:
                logging.info("Completed %s of unknown", count)
    filename = "species_per_square.json"
    set_neighbours(grid_meta)
    logging.info("Writing metadata to %s", filename)
    metadata = {
        "latest_obs_date": latest_date.isoformat(),
        "generated": datetime.now().isoformat(),
        "source": args.csv.name,
        "grid_meta": grid_meta,
    }
    with open(filename, "w") as file:
        json.dump(metadata, file, indent=4)


def add_new_square(squares, lng, lat, region_meta):
    bounds = [
        lng - MAX_LNG / 2,
        lat - MAX_LAT / 2,
        lng + MAX_LNG / 2,
        lat + MAX_LAT / 2,
    ]
    logging.info(
        "Adding new square for %s %s %s maps %s",
        lng,
        lat,
        bounds,
        f"https://maps.google.com/?q={lat},{lng}",
    )
    _, region_bound = find_region_meta(region_meta, lng, lat)
    region_code = None
    if region_bound is not None:
        region_code = region_bound["code"]
    meta = {"region_code": region_code, "bounds": bounds, "species_per_month": {}}
    squares.append(meta)
    squares = sorted(squares, key=lambda square: square["bounds"][0])
    return squares, meta


def find_square(square_meta, lng, lat):
    for i, meta in enumerate(square_meta):
        square = meta["bounds"]
        if (
            lng >= square[0]
            and lng <= square[2]
            and lat >= square[1]
            and lat <= square[3]
        ):
            # print("Found square for ", lng, lat, " as ",square)
            return i, square
    return 0, None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="CSV file toload")
    args = parser.parse_args()
    args.csv = Path(args.csv)
    return args


if __name__ == "__main__":
    main()
