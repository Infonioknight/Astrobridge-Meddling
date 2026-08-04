import lsdb
import pandas as pd
from lsdb.streams import CatalogStream

RADIUS_ARCSEC = 1.0

targets = pd.read_csv("targets_for_crossmatch.csv")

target_cat = lsdb.from_dataframe(
    targets,
    ra_column="ra_in",
    dec_column="dec_in",
    margin_threshold=5.0,
)

legacy = lsdb.open_catalog(
    "hf://datasets/UniverseTBD/mmu_ssl_legacysurvey_north",
    columns=["object_id", "ra", "dec"],
)

xm = target_cat.crossmatch(legacy, radius_arcsec=RADIUS_ARCSEC, n_neighbors=1)

batch = [chunk for chunk in CatalogStream(catalog=xm) if len(chunk)]

matches = pd.concat(batch).reset_index() if batch else pd.DataFrame()
if len(matches):
    matches.to_csv("literature_x_legacysurvey_matches.csv", index=False)
    print(f"{len(matches)} / {len(targets)} targets matched ({len(matches)/len(targets):.1%})")
    print(matches["_dist_arcsec"].describe())
else:
    print("no matches")