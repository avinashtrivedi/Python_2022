import datetime
import argparse
import pandas as pd
from .sendgrid import sendgrid_features
from .listrak import listrak_features
from .pageview import pageview_features
from .dbhist import db_features
from .subcycle import cycle_features
from .utils import eow_date


def checking_missing(data, ds, dl, dp, dc):
    refset = set([(c, d) for c, d in data[["customer_id", "ref_date"]].values])
    for name, other_ds in zip(("sendgrid", "listrak", "pageview", "cycle"), (ds, dl, dp, dc)):
        miss = refset - set([(c, d) for c, d in other_ds[["customer_id", "ref_date"]].values])
        if len(miss) > 0:
            print(f"Missing data entries in {name}:", miss)
    if "target" in data.columns:
        nulls = data.target.isnull().sum()
        if nulls:
            print("Nulls in the target variable:", nulls)


def generate_features(
    grid,
    n_weeks=4,
    ds=None,
    dl=None,
    dp=None,
    db=None,
    dc=None,
    datapath=None,
    cfg=None,
):
    """returns the features and targets based on a grid input"""

    if ds is None:
        ds = sendgrid_features(grid, n_weeks=n_weeks)
    if dl is None:
        dl = listrak_features(grid, n_weeks=n_weeks)
    if dp is None:
        dp = pageview_features(grid, n_weeks=n_weeks)
    if db is None:
        db = db_features(grid, datapath=datapath, cfg=cfg)
    if dc is None:
        dc = cycle_features(grid)

    data = ds.merge(dl, on=["customer_id", "ref_date"], how="outer")
    data = data.merge(dp, on=["customer_id", "ref_date"], how="outer")
    data = data.merge(db, on=["customer_id", "ref_date"], how="outer")
    data = data.merge(dc, on=["customer_id", "ref_date"], how="inner")

    # add the target to the output if available in the grid
    if "target" in grid.columns:
        tdf = grid[["customer_id", "ref_date", "target"]].copy()
        # convert the date time to end of week date
        tdf.ref_date = tdf.ref_date.apply(eow_date).apply(
            lambda d: str(datetime.date(d.year, d.month, d.day))
        )
        data = data.merge(tdf, on=["customer_id", "ref_date"], how="left")

    checking_missing(data, ds, dl, dp, dc)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features for model training")
    parser.add_argument(
        "gridfile",
        type=str,
        help="path to the grid file with customer ids and reference dates",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="customerrisk_features.csv",
        help="full path to save the features CSV file",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default=None,
        help="path to data files to use as db cache (optional)",
    )
    args = parser.parse_args()
    grid = pd.read_csv(args.gridfile)
    grid.ref_date = pd.to_datetime(grid.ref_date)
    datafeat = generate_features(grid, datapath=args.datapath)
    datafeat.to_csv(args.output, index=False)
