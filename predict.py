import datetime
from pathlib import Path
from sklearn import metrics
import numpy as np
import pandas as pd
import lightgbm as lgb
from dateutil.parser import parse
from .features import generate_features
from humpredict.core import get_db_connection, AttrDict


def get_default_modelpath():
    basepath = Path(__file__).absolute().parent.parent.parent
    return str(basepath / "models" / "production.bqrisk.txt")


def predict_grid(grid, modelpath=None, cfg=None):
    if modelpath is None:
        modelpath = get_default_modelpath()
    assert all(c in grid.columns for c in ("customer_id", "ref_date"))
    data = generate_features(grid, cfg=cfg)
    non_feat_cols = ["customer_id", "ref_date"]
    if "target" in grid.columns:
        non_feat_cols.append("target")
        evaluate = True
    else:
        evaluate = False
    Xtest = data.drop(non_feat_cols, axis=1)
    gbm = lgb.Booster(model_file=modelpath)
    ypred = gbm.predict(Xtest)
    if evaluate:
        f1score = metrics.f1_score(data.target, np.where(ypred > 0.5, 1, 0))
        recall = metrics.recall_score(data.target, np.where(ypred > 0.5, 1, 0))
        print(f"F1 {f1score}, recall {recall}")
    return data.customer_id.values.tolist(), ypred


def predict(customer_id, ref_date=None, modelpath=None, version="production", cfg=None):
    if modelpath is None:
        modelpath = get_default_modelpath()
    if ref_date is None:
        ref_date = datetime.datetime.now()
    grid = pd.DataFrame([{"customer_id": customer_id, "ref_date": ref_date}])
    _, ypred = predict_grid(grid, modelpath, cfg=cfg)
    return ypred[0]


def fromisoformat(ref_date):
    """this function is needed because Python 3.6 does not have
    datetime.fromisoformat. When we upgrade we can replace this with
    code from the standard lib
    """
    return parse(ref_date)


def active_subscription_ids(ref_date, cfg, month_range=4):
    """Return the active subscribers ids considering the isActive flag and only ids
    that have at least 1 invoice in the last N months
    """
    # FIXME
    if cfg is None:
        cfg = AttrDict(
            {"DBURL": "mysql+mysqlconnector://hum_dash:dash8228@104.197.102.90/hum_data"}
        )
    con = get_db_connection(cfg)
    # active subscribers
    subs = pd.read_sql_query("select customer_id, isActive, nextCycleDate from Subscription where isActive = 1", con)
    # clients with an invoice in the last 6 months
    if ref_date is None:
        ref_date = datetime.datetime.now() - datetime.timedelta(month_range*31)
    elif type(ref_date) == str:
        ref_date = fromisoformat(ref_date)
    inv = pd.read_sql_query(f"select customer_id from Invoice where createdAt >=  '{ref_date.isoformat()}'", con)
    return list(set(subs.customer_id.values) & set(inv.customer_id.values))


def predictlist(ids, ref_date=None, modelpath=None, version="production", cfg=None):
    """predict for a list of customer ids; if the reference date is not given it will
    consider the current date
    """
    if modelpath is None:
        modelpath = get_default_modelpath()
    if ref_date is None:
        ref_date = datetime.datetime.now()
    elif type(ref_date) == str:
        ref_date = fromisoformat(ref_date)

    grid = pd.DataFrame([{"customer_id": v, "ref_date": ref_date} for v in ids])
    ids, ypred = predict_grid(grid, modelpath, cfg=cfg)
    res = [{"customer_id": cid, "proba": float(proba)} for cid, proba in zip(ids, ypred)]
    return set_risk_levels(res)


def predictfull(limit=0, proba=None, ref_date=None, modelpath=None, version="production",
                sort_ascending=False, cfg=None):
    """ predict for all customer ids in scope for subscription
    """
    ids = active_subscription_ids(ref_date, cfg)
    res = predictlist(ids, ref_date=ref_date, modelpath=modelpath, version=version,
                      cfg=cfg)
    # if probability is given, filter by probability
    if proba is not None:
        res = [itm for itm in res if itm["proba"] >= proba]
    elif limit > 0:
        do_reverse = False if sort_ascending else True
        res = list(sorted(res, reverse=do_reverse, key=lambda x: x["proba"]))[:limit]
    return set_risk_levels(res)


def set_risk_levels(results):
    res = results.copy()
    for itm in res:
        if itm["proba"] < 0.2:
            itm["cancel_risk_level"] = "L"
        elif itm["proba"] < 0.7:
            itm["cancel_risk_level"] = "M"
        else:
            itm["cancel_risk_level"] = "H"
    return res
