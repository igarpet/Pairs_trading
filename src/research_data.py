"""Formation-only data preparation."""

import pandas as pd


def normalize_frame(frame):
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    frame.columns = frame.columns.astype(str)
    return frame.sort_index()


def prepare_prices(prices, config, membership=None, allow_legacy=False):
    p = normalize_frame(prices).sort_index(axis=1)
    cut = int(len(p) * config.train_fraction)
    formation_end = p.index[cut - 1]

    if membership is not None:
        m = membership.copy()
        for column in ["member_from", "member_to", "known_at"]:
            m[column] = pd.to_datetime(m[column])
        keep = m.loc[
            (m.member_from <= formation_end)
            & (m.known_at <= formation_end)
            & (m.member_to.isna() | (m.member_to > formation_end)),
            "ticker",
        ].astype(str)
        p = p.loc[:, sorted(set(keep))]

    train = p.iloc[:cut].copy()
    selected = (train.isna().mean() <= config.max_missing_fraction) & train.iloc[0].notna()
    retained = list(train.columns[selected])
    report = pd.DataFrame(
        {
            "ticker": train.columns,
            "formation_missing_fraction": train.isna().mean().values,
            "retained": selected.values,
        }
    )

    train = train[retained].ffill()
    test = p.iloc[cut:][retained]
    return train, test, report


def read_series(path):
    frame = normalize_frame(pd.read_parquet(path))
    return frame.iloc[:, 0].astype(float)
