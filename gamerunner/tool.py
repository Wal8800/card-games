import os
from typing import Dict, Any

import yaml

import pandas as pd
from tqdm import tqdm


def convert_config_csv_to_yml():
    for root, dirs, files in tqdm(os.walk("./experiments")):
        if "config.csv" not in files:
            continue

        df = pd.read_csv(f"{root}/config.csv")

        data: Dict[Any, Any] = df.to_dict(orient="records")[0]
        with open(f"{root}/config.yaml", "w") as yml_file:
            yaml.dump(data, yml_file)


if __name__ == "__main__":
    convert_config_csv_to_yml()
