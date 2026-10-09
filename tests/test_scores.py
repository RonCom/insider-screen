import numpy as np
import pandas as pd
import pytest

from insider_screen import scores as sc


def events(n=600, charged=12, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "event_id": [f"e{i}" for i in range(n)],
        "dollar_volume": rng.lognormal(15, 2, n),
        "sic": rng.choice(["2834", "6022", "7372", "1311"], n),
        "abn_volume": rng.normal(0, 0.6, n), "last5_share": rng.normal(0.3, 0.1, n),
        "scar": rng.normal(0, 1, n), "short_share_z": rng.normal(0, 2, n)})
    y = pd.Series(0, index=df.index)
    y.iloc[:charged] = 1
    df.loc[:charged - 1, ["abn_volume", "scar", "short_share_z"]] += [2.0, 3.0, 4.0]
    return df, y


def test_sic_division():
    assert sc.sic_division("2834") == "D" and sc.sic_division("6022") == "H" and sc.sic_division(None) == "?"


def test_peer_groups_fall_back_when_small():
    df, _ = events(n=60)
    out = sc.peer_z(df, min_peers=20)
    sizes = out.groupby("peer_group").size()
    # 60 events in 5 x 4 cells: no size-by-sector cell reaches 20, so all fall back to their size quintile
    assert all(g.endswith("/all") or g == "all" for g in sizes.index)
    assert out.z_scar.notna().all()


def test_composite_ranks_planted_events_on_top():
    df, y = events()
    z = sc.peer_z(df)
    s = sc.composite(z)
    l5, hit, k = sc.lift(s, y, 0.05)
    assert hit >= 10 and l5 > 10
    assert sc.auc(s, y) > 0.95


def test_supervised_is_out_of_fold():
    df, y = events()
    z = sc.peer_z(df)
    s = sc.supervised_oof(z, y)
    assert s.notna().all() and sc.auc(s, y) > 0.8


def test_lift_counts():
    s = pd.Series([5, 4, 3, 2, 1, 0, 0, 0, 0, 0] * 2, dtype=float)
    y = pd.Series([1] + [0] * 19)
    assert sc.lift(s, y, 0.05) == (pytest.approx(20.0), 1, 1)
