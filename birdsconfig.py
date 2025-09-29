# when training the keys here will be relabeld to the values, can be used to merge multiple species into a single label
# i.e red and yellow parekeets into parakeet
RELABEL_MAP = {
    "gragoo1": "cangoo",
    "x00458": "duck",
    "mallar3": "duck",
    "domesz3tic-chicken": "redjun1",
    "yeepen1": "litpen1",
    "yefpar3": "parake",
    "refpar4": "parake",
    "nezrob2": "nezrob3",
    "redpol1": "comred",
    "blbgul1": "gull",
    "silgul2": "gull",
    "kelgul": "gull",
    "larus": "gull",
    "rebgul1": "gull",
    "new zealand dotterel": "dobplo1",
    "rebdot1": "dobplo1",
}


# add labels you want regardless of number of samples
# birds from all birds will be added if they have over 50 records and 2 validation samples
BIRD_TRAIN_LABELS = [
    "ausbit1",
    "ausmag2",
    "baicra4",
    "blkswa",
    "eurbla",
    "fernbi1",
    "gryger1",
    "houspa",
    "kiwi",
    "morepo2",
    "nezfan1",
    "redjun",
    "silver3",
    "sonthr1",
    "tui1",
    "y01193",
]
from utils import get_all_birds

ALL_BIRDS = get_all_birds()
ALL_BIRDS.append("bird")
ALL_BIRDS.append("gull")
ALL_BIRDS.append("dove")
ALL_BIRDS.append("kiwi")
ALL_BIRDS.append("norfolk silkeye")
ALL_BIRDS.append("tree weta")


# NOISE SECTION
NOISE_LABELS = [
    "Acoustic_guitar",
    "Bass_guitar",
    "Clapping",
    "Coin_(dropping)",
    "Crash_cymbal",
    "Dishes_and_pots_and_pans",
    "Engine",
    "Fart",
    "Fire",
    "Fireworks",
    "Glass",
    "Hi-hat",
    "Piano",
    "Rain",
    "Slam",
    "Squeak",
    "Tearing",
    "Walk_or_footsteps",
    "Wind",
    "Writing",
    "airplane",
    "beach",
    "breathing",
    "brown",
    "brushing_teeth",
    "campfire",
    "can_opening",
    "car_horn",
    "chainsaw",
    "chirping_birds",
    "church_bells",
    "city",
    "clapping",
    "clock_alarm",
    "clock_tick",
    "construction",
    "crackling_fire",
    "door_wood_creaks",
    "door_wood_knock",
    "drinking_sipping",
    "engine",
    "factory",
    "fan",
    "fireworks",
    "footsteps",
    "forest",
    "glass_breaking",
    "hand_saw",
    "helicopter",
    "keyboard_typing",
    "library",
    "mouse_click",
    "noise",
    "planecabin",
    "pouring_water",
    "rain",
    "rainforest",
    "river",
    "sea_waves",
    "siren",
    "snoring",
    "starship",
    "static",
    "thunderstorm",
    "toilet_flush",
    "train",
    "vacuum_cleaner",
    "vehicle",
    "washing_machine",
    "water",
    "water_drops",
    "white",
    "wind",
]

ANIMAL_LABELS = [
    "bear",
    "cat",
    "chicken",
    "cow",
    "dog",
    "dolphin",
    "donkey",
    "elephant",
    "frog",
    "hen",
    "horse",
    "lion",
    "monkey",
    "pig",
    "possum",
    "rodent",
    "sheep",
]

INSECT_LABELS = ["crickets", "insects"]

HUMAN_LABELS = [
    "Breathing",
    "Coughing",
    "Crying baby",
    "Laughing",
    "Sneezing",
    "coughing",
    "crying_baby",
    "laughing",
    "sneezing",
]


EXTRA_LABELS = ["rooster", "frog", "insect", "human", "noise"]
OTHER_LABELS = []


for l in NOISE_LABELS:
    if l != "noise":
        RELABEL_MAP[l] = "noise"


for l in HUMAN_LABELS:
    if l != "human":
        RELABEL_MAP[l] = "human"


for l in OTHER_LABELS:
    if l != "other":
        RELABEL_MAP[l] = "other"
