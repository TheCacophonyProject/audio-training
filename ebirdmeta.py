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
import matplotlib.pyplot as plt

NZ_RECT = [166.419, -34.0, 178.517093541, -47.29]

# maximum width height from kml file
MAX_LNG=  0.15556536514500863  
MAX_LAT = 0.10025559492370206

# chatham??
# RECT_WIDTH = 0.2
def read_squares():

    import geopandas as gpd
    global MAX_LNG,MAX_LAT

    kml_file = "Atlas Grid Squares with names_June 2020.kml"
    gdf = gpd.read_file(kml_file, driver="KML")
    all_bounds = []
    max_lng = None
    max_lat = None
    for geo in gdf["geometry"]:
        all_bounds.append(geo.bounds)
        lng = geo.bounds[2]-geo.bounds[0]
        lat = geo.bounds[3]-geo.bounds[1]
        if max_lng is None or lng > max_lng:
            max_lng =lng
        if max_lat is None or lat > max_lat:
            max_lat = lat
    MAX_LNG = max_lng
    MAX_LAT = max_lat
    # print("lng length", geo.bounds[2]-geo.bounds[0],"lat width", geo.bounds[3]-geo.bounds[1])
    return all_bounds
    # import folium

    # # Create a map centered at a specific location
    # m = folium.Map(location=[NZ_RECT[1],NZ_RECT[0]], zoom_start=2)
    # folium.Rectangle(
    #     bounds=[[NZ_RECT[1],NZ_RECT[0]], [NZ_RECT[3], NZ_RECT[2]]],
    #     line_join="round",
    #     dash_array="5, 5",
    # ).add_to(m)

    # min_lng = np.amin([ b[0] for b in all_bounds ])
    # print("MIn max 0",np.amin([ b[0] for b in all_bounds ]),np.amax([ b[0] for b in all_bounds ]))
    # print("MIn max 2",np.amin([ b[2] for b in all_bounds ]),np.amax([ b[2] for b in all_bounds ]))

    # print("MIn max 1",np.amin([ b[1] for b in all_bounds ]),np.amax([ b[1] for b in all_bounds ]))
    # print("MIn max 3",np.amin([ b[3] for b in all_bounds ]),np.amax([ b[3] for b in all_bounds ]))

    # max_lng = np.amax([ b[2] for b in all_bounds ])

    # min_lat = np.amin([ b[1] for b in all_bounds ])
    # max_lat = np.amax([ b[3] for b in all_bounds ])
    # print("Lng",min_lng,max_lng, "lat",min_lat,max_lat)
    # print(gdf["geometry"])
    # return
    # geojson_data = gdf.to_json()
    # print(geojson_data)
    # folium.GeoJson(geojson_data).add_to(m)

    # Add a marker
    # folium.Marker([NZ_RECT[3],NZ_RECT[2]], popup="bottom right").add_to(m)
    # folium.Marker([NZ_RECT[1],NZ_RECT[0]], popup="top left").add_to(m)

    # Save the map as an HTML file or display it
    # width = abs(NZ_RECT[2] - NZ_RECT[0])
    # squares_wide = int(math.ceil(width / RECT_WIDTH))

    # height = abs(NZ_RECT[3] - NZ_RECT[1])
    # print("Hieght is ",height,width)
    # squares_high = int(math.ceil(height / RECT_WIDTH))
    # for row_i in range(squares_high):
    #     row= NZ_RECT[3] - row_i * RECT_WIDTH

    #     for col_i in range(squares_wide):
    #         row= NZ_RECT[3] + row_i * RECT_WIDTH
    #         col =  NZ_RECT[0] + col_i * RECT_WIDTH

    #         # print("Adding ", [col,row], [col+RECT_WIDTH, row+RECT_WIDTH])
    #         folium.Rectangle(
    #             bounds=[[row,col],[row+RECT_WIDTH, col+RECT_WIDTH]],
    #             line_join="round",
    #             dash_array="5, 5",
    #         ).add_to(m)
    # break
    # break
    # m.save("Test.html")
    # return


def common_to_ebird():
    ebird_labels = {}
    with open("eBird_taxonomy_v2024.csv") as f:
        f.readline()

        for line in f:
            split_l = line.split(",")

            ebird_labels[split_l[4].lower()] = split_l[2]
            # print("ADding ",split_l[9], " to " , split_l[2])
            # if (
            #     split_l[1].lower() in test_labels
            #     or split_l[9].lower() in test_labels
            #     or "kiwi" in split_l[1]
            # ):
            #     ebird_labels.add(split_l[2])
    return ebird_labels


def species_per_loc(squares, lng, lat):
    species_json = Path("species_per_square.json")
    with species_json.open("r") as f:
        species_meta = json.load(f)
    i, square = find_square(squares, lng, lat)
    print("Found ", i, square)
    print("Spcies count ", species_meta[i])
    return species_meta[1]


def species_per_lbound(squares, region_bound):
    species_json = Path("species_per_square.json")
    with species_json.open("r") as f:
        species_meta = json.load(f)

    square_species = set()
    total_matches = 0
    print("Checking bound", region_bound)
    for i, square in enumerate(squares):
        corners = []
        # y is lat
        # lng,lat
        corners.append((square[0], square[1]))

        corners.append((square[0], square[3]))
        corners.append((square[2], square[1]))
        corners.append((square[2], square[3]))
        for lng, lat in corners:
            # print("Checking ",lng, " against ",square[0])
            if (
                lng >= region_bound["minX"]
                and lng <= region_bound["maxX"]
                and lat >= region_bound["minY"]
                and lat <= region_bound["maxY"]
            ):
                total_matches += 1
                square_species.update(list(species_meta[i].keys()))
                # i, square = find_square(squares, lng, lat)
                # 1/0
                break

    print("Matches are ", total_matches)
    return square_species


def find_in_old(big_grid_meta, lng, lat):

    #     grid_file = Path("ebird_species.json")
    #     with grid_file.open("r") as f:
    #         big_grid_meta = json.load(f)
    for code, species_info in big_grid_meta.items():
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

def find_fast(squares, lng, lat):
    high = len(squares)
    low = 0
    found = None
    # squares in order of lng so can binary search
    while high > low:
        mid = (high + low) // 2
        # print("Checing",high,low,mid)
        # if mid >= len(squares):
            # return None
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
            return mid,square
        mid += 1

    mid = found - 1
  # check back until out of lng or found match
    while mid >=0:
        square = squares[mid]
        bounds = square["bounds"]
        if bounds[0] > lng:
            break
        if bounds[1] <= lat and bounds[3] >= lat:
            return mid,square
        mid -= 1
    logging.error("Could not find species square for %s, %s", lng, lat)    
    return None



def set_neighbours(squares):
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

    print("Max width ", max_width, max_height)


def species_per_month_nz():

    filename = "species_per_squarev3.json"

    with open(filename, "r") as file:
        species_meta = json.load(file)

    common_ebird_map = common_to_ebird()
    for grid in species_meta:
        # print("Checking ",grid)
        ebird_dict = {}
        for species, months in grid["species_per_month"].items():
            ebird_id = common_ebird_map[species.lower()]
            # print(ebird_id, species)
            ebird_dict[ebird_id] = months
        grid["species_per_month"] = ebird_dict
    filename = "species_per_squarev4.json"

    with open(filename, "w") as file:
        species_meta = json.dump(species_meta, file)
    # 1/0
    # 1 / 0
    # sorted_by_lng = sorted(species_meta,key=lambda square: square["bounds"][0])
    # set_neighbours(sorted_by_lng)
    # filename = "species_per_squarev3.json"

    # with open(filename, "w") as file:
    #     species_meta = json.dump(sorted_by_lng, file)
    # 1/0
    chch_lat = -43.52048909400715
    chch_lng = 172.6223369746072

    fast_s = find_square(species_meta, chch_lng, chch_lat)
    print("Neighbours are ", fast_s["neighbours_i"])
    species_per_month = fast_s["species_per_month"]
    for neighbour in fast_s["neighbours_i"]:
        continue
        neighbour_species = species_meta[neighbour]["species_per_month"]
        for species, month_data in neighbour_species.items():
            total = sum(month_data.values())
            print(f"{species}={total}")
            if species not in species_per_month:
                species_per_month[species] = month_data.copy()
                continue
            for (
                m,
                c,
            ) in month_data.items():
                species_per_month[species][m] += c
            # print("Merge",species)
    print("MERGED")
    print("")
    abs_total = 0
    max_obs = None
    max_species = None
    for species, month in species_per_month.items():
        total = sum(month.values())
        if max_obs is None or total > max_obs:
            max_obs = total
            max_species = species

    for species, month in species_per_month.items():
        total = sum(month.values())
        abs_total += total
        print(f"{species}={total} , {round(100*total/max_obs,1)}%")
    print(
        len(species_per_month),
        " species ",
        abs_total,
        len(species_per_month),
        "Max obs is ",
        max_obs,
        max_species,
    )
    1 / 0
    print("Fast s ", fast_s["bounds"])
    print(fast_s["species_per_month"])
    i, square = slow_find(species_meta, chch_lng, chch_lat)
    print("Slow is ", square["bounds"])
    1 / 0
    bird = "Little Penguin"
    global_per_month = {}
    per_lat = {}
    for square in species_meta:
        lng = square["bounds"][0]
        if lng in per_lat:
            per_lat[lng] += 1
            print("DOUBLEd", lng)
        else:
            per_lat[lng] = 1
        species_per_month = square["species_per_month"]
        for species, month_count in species_per_month.items():
            if species != bird:
                continue
            global_month = global_per_month.get(species)
            if global_month is None:
                print("Not got ", species)
                global_month = {}
                for i in range(12):
                    global_month[i + 1] = 0
                global_per_month[species] = global_month
            for m_i, count in month_count.items():

                global_month[int(m_i)] += count
    print(per_lat)
    lats = list(per_lat.keys())
    lats.sort()
    print(lats)
    return
    print("Species per month is ", global_per_month)
    # x = np.arange(12)+1
    import calendar

    x = list(calendar.month_abbr)[1:]
    y = global_per_month[bird].values()

    plt.plot(x, y)
    plt.show()
    # json.dump(species, file, indent=4)


def main():
    # squares = read_squares()

    # species_per_month_nz()
    # # return
    # grid_file = Path("ebird_species.json")
    # with grid_file.open("r") as f:
    #     big_grid_meta = json.load(f)

    # squares = read_squares()
    # akl_lat = -36.85545854963461
    # akl_lng = 174.68532793372913
    # # chch_lat = -43.52048909400715
    # # chch_lng = 172.6223369746072
    # # print("Square width is ",squares[0][2] - squares[0][0])
    # # return

    # grid_file = Path("ebird_species.json")
    # with grid_file.open("r") as f:
    #     big_grid_meta = json.load(f)
    # old_species, bound = find_in_old(big_grid_meta,akl_lng, akl_lat)

    # species_meta = species_per_lbound(squares, bound["info"]["bounds"])
    # # species_meta = species_per_loc(squares, chch_lng, chch_lat)

    # small_grid_ebirds = []
    # for species in species_meta:
    #     small_grid_ebirds.append(common_ebird_map[species.lower()])
    #     # print("Species ", species, common_ebird_map.get(species.lower(),"NONE"))
    # for old in old_species:
    #     if old not in small_grid_ebirds:
    #         common = ""
    #         for k, v in common_ebird_map.items():
    #             if v == old:
    #                 common = k
    #                 break
    #         print("Missing ", common, " which is ", old)
    # # return

    # species = []
    # for square in squares:
    #     lng = (square[2] + square[0]) / 2
    #     lat = (square[1] + square[3]) / 2

    #     _, region_bound = find_in_old(big_grid_meta, lng, lat)
    #     region_code = None
    #     if region_bound is None:
    #         region_code = region_code["code"]
    #         continue
    #     # print("BOUND is ",region_bound)
    #     # print("square is ",square)

    #     meta = {"region_code": region_code, "bounds": square, "species_per_month": {}}

    #     species.append(meta)
    # return
    fmt = "%(process)d %(thread)s:%(levelname)7s %(message)s"
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S"
    )
    args = parse_args()

    squares = read_squares()
    squares = sorted(squares,key=lambda square: square[0])

    grid_file = Path("ebird_species.json")
    with grid_file.open("r") as f:
        big_grid_meta = json.load(f)
    grid_meta = []
    for square in squares:
        lng = (square[2] + square[0]) / 2
        lat = (square[1] + square[3]) / 2

        _, region_bound = find_in_old(big_grid_meta, lng, lat)
        region_code = None
        if region_bound is not  None:
            region_code = region_bound["code"]

            # continue
        # print("BOUND is ",region_bound)
        # print("square is ",square)

        meta = {"region_code": region_code, "bounds": square, "species_per_month": {}}

        grid_meta.append(meta)
    common_ebird_map = common_to_ebird()

    # species =
    with open("species_per_square.json", "r") as f:
        species = json.load(f)
    
    print("Loading ", args.csv)
    count = 0
    by_common_name = {}

    with args.csv.open("r") as f:
        dreader = csv.reader(f, delimiter="\t", quotechar="|")
        first = True
        headers = None
        atlas_block_i = None
        name_i = None
        lat_i = None
        lng_i = None
        type_i = None
        date_i = None
        for row in dreader:
            count += 1
            if first:
                headers = row
                print(headers)
                atlas_block_i = headers.index("ATLAS BLOCK")
                name_i = headers.index("COMMON NAME")
                lat_i = headers.index("LATITUDE")
                lng_i = headers.index("LONGITUDE")
                date_i = headers.index("OBSERVATION DATE")
                type_i = headers.index("OBSERVATION TYPE")
                county_i = headers.index("COUNTY")
                first = False
                continue
            lat = float(row[lat_i])
            lng = float(row[lng_i])
            res =  find_fast(grid_meta, lng, lat)
            if res is None:
                # shall we just add these
                grid_meta, square = add_new_square(grid_meta,lng,lat,big_grid_meta)
                # print("Could not find match for ", lat, lng,row)
                # print("Res 2 is ", res2)   
                # break             
                # continue
            else:
                _,square = res
            
            common_name = row[name_i]
            ebird_id = common_ebird_map.get(common_name.lower(),"unknown")
           

            obs_type = row[type_i]
            obs_date = parse_date(row[date_i])
            species_dic = square["species_per_month"]
            if ebird_id not in species_dic:

                month_count = {}
                for i in range(12):
                    month_count[i + 1] = 0
                species_dic[ebird_id] = month_count
            species_dic[ebird_id][obs_date.month] += 1
            # if count > 10:
                # break
            # break
    filename = "species_per_square.json"

    with open(filename, "w") as file:
        json.dump(grid_meta, file, indent=4)


def add_new_square(squares, lng,lat,region_meta):
    bounds = [lng - MAX_LNG / 2, lat - MAX_LAT / 2, lng + MAX_LNG / 2 , lat + MAX_LAT / 2]
    logging.info("Adding new square for %s %s %s", lng,lat, bounds)
    _,region_bound = find_in_old(region_meta,lng,lat)
    region_code = None
    if region_bound is not None:
        region_code = region_bound["code"]
    meta = {"region_code": region_code, "bounds": bounds, "species_per_month": {}}
    squares.append(meta)
    squares = sorted(squares,key=lambda square: square["bounds"][0])
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
