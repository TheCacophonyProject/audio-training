import requests
import time
import json


# save ebird species list per region of nz
key = "YOUR KEY"
api_url = "https://api.ebird.org/v2/product/spplist/{}"


# taxonomy = "https://api.ebird.org/v2/ref/taxonomy/ebird"
# headers = {'x-ebirdapitoken': key}
# r = requests.get(taxonomy,headers=headers)
# r.raise_for_status()
# # taxonomy = r.json()
# # print(taxonomy)
# with open("ebird_taxonomy.json", "w") as f:
#     f.write(r.text)

regions_url = "https://api.ebird.org/v2/ref/region/list/subnational1/NZ"
headers = {"x-ebirdapitoken": key}
r = requests.get(regions_url, headers=headers)
r.raise_for_status()
regions = r.json()


regions_url = "https://api.ebird.org/v2/ref/region/info/NF"
r = requests.get(regions_url, headers=headers)
r.raise_for_status()
nf_region = r.json()
regions.append(nf_region)
nz_birds = {}

region_info_url = "https://api.ebird.org/v2/ref/region/info/{}"
for region in regions:
    country_code = region.get("code")

    url = region_info_url.format(country_code)
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    region_info = r.json()
    print(region_info)
    region["info"] = region_info
    url = api_url.format(country_code)
    r = requests.get(url, headers=headers)
    r.raise_for_status()

    species = r.json()
    nz_birds[country_code] = {
        "region": region,
        "species": species,
        "updatedAt": time.time(),
    }
    # break
out_file = "ebird_species.json"
with open(out_file, "w") as f:
    json.dump(nz_birds, f, indent=4)
