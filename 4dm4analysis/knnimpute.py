import sqlite3
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from utils import Dataset
from utils import csv_to_sql

interested_rounds = ["Q", "RO32", "RO16", "QF", "SF", "F", "GF"]
interested_types = ["LN", "HB", "RC", "TB"]
_4dm = Dataset("datasets/4dm4.db")
old_ds = _4dm.get_old_dataset(interested_rounds, interested_types, True)
played = pd.notna(old_ds)
mean = _4dm.query(
    f"SELECT player_name, avg(score_logit) as average_score from scores where round in {tuple(interested_rounds)} and beatmap_type in {tuple(interested_types)} GROUP BY player_name"
)

for player_name in mean["player_name"]:
    old_ds.loc[player_name] -= float(
        mean[mean["player_name"] == player_name]["average_score"]
    )

knn = KNNImputer(n_neighbors=2)
res = knn.fit_transform(old_ds)

new_ds = pd.DataFrame(res, index=old_ds.index, columns=old_ds.columns)
for player_name in mean["player_name"]:
    new_ds.loc[player_name] += float(
        mean[mean["player_name"] == player_name]["average_score"]
    )

new_ds = new_ds.apply(lambda x: 1000000 / (1 + np.exp(-x)))
new_sql = sqlite3.connect("4dm4_impute.db")
cursor = new_sql.cursor()

csv_to_sql(cursor, new_ds, played)
new_sql.commit()
