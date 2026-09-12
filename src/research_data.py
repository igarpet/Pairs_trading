import pandas as pd


def prepare_prices(prices, config, membership=None):
    p = prices.copy()
    p.index = pd.to_datetime(p.index).tz_localize(None).normalize()
    p.columns = p.columns.astype(str)
    p = p.sort_index().sort_index(axis=1)

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
    retained = train.columns[selected]

    availability = pd.DataFrame(
        {
            "ticker": train.columns,
            "formation_missing_fraction": train.isna().mean().values,
            "retained": selected.values,
        }
    )

    train = train[retained].ffill()
    test = p.iloc[cut:][retained]
    return train, test, availability
