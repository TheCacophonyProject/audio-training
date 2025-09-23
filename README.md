# Audio Model

## Ebird Grid data

`python3 ebirdgrid.py <path to ebird csv data>`
This will generate a json file (species_per_square.json) containing metadata on species observed in each roughtly 10 by 10 km Square in New Zealand.

For each square the metadata will give a count of species per month of the year.

This file is used by audio model predictions to filter out predictions on species that haven't been observed in a grid
```
{
    "latest_obs_date": "2025-03-31T00:00:00",
    "generated": "2025-08-20T11:14:46.052601",
    "source": "ebirdmeta.csv",
    "grid_meta": [
        {
            "region_code": null,
            "bounds": [
                -180.0676871825725,
                -44.819371497461844,
                -179.91212181742748,
                -44.71911590253815
            ],
            "species_per_month": {
                "brbpri1": {
                    "1": 0,
                    "2": 0,
                    "3": 0,
                    "4": 0,
                    "5": 0,
                    "6": 0,
                    "7": 0,
                    "8": 0,
                    "9": 0,
                    "10": 0,
                    "11": 0,
                    "12": 0,
                },
            }
        }
    ]
}
```