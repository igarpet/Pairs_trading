"""Input validation and formation-only cleaning."""
import numpy as np
import pandas as pd


def normalize_frame(frame):
    frame = frame.copy()
    frame.index=pd.to_datetime(frame.index).tz_localize(None).normalize()
    frame.columns=frame.columns.astype(str)
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError('Duplicate dates or columns.')
    return frame.sort_index()


def prepare_prices(prices, config, membership=None, allow_legacy=False):
    p=normalize_frame(prices).sort_index(axis=1)
    cut=int(len(p)*config.train_fraction)
    if cut < max(100,config.memory_window+1) or len(p)-cut < 3:
        raise ValueError('Insufficient formation or OOS observations.')
    formation_end=p.index[cut-1]
    if membership is None and not allow_legacy:
        raise ValueError('Supply --membership or explicitly acknowledge --allow-legacy-universe.')
    if membership is not None:
        required={'ticker','member_from','member_to','known_at'}
        if not required.issubset(membership): raise ValueError(f'Membership requires {sorted(required)}')
        m=membership.copy()
        for c in ['member_from','member_to','known_at']: m[c]=pd.to_datetime(m[c])
        if m[['member_from','known_at']].isna().any().any(): raise ValueError('Membership dates cannot be missing.')
        keep=m.loc[(m.member_from<=formation_end)&(m.known_at<=formation_end)&(m.member_to.isna()|(m.member_to>formation_end)),'ticker'].astype(str)
        missing=set(keep)-set(p.columns)
        if missing: raise ValueError(f'Price input missing dated members: {sorted(missing)}')
        p=p.loc[:,sorted(set(keep))]
    train=p.iloc[:cut].copy()
    selected=(train.isna().mean()<=config.max_missing_fraction)&train.iloc[0].notna()
    retained=list(train.columns[selected])
    report=pd.DataFrame({'ticker':train.columns,'formation_missing_fraction':train.isna().mean().values,
                         'retained':selected.values})
    train=train[retained].ffill()  # prior-only fill on formation; no backfill, no OOS filling
    test=p.iloc[cut:][retained]
    if len(retained)<2: raise ValueError('Fewer than two assets pass formation-only availability rules.')
    for frame in [train,test]:
        if not np.isfinite(frame.to_numpy()).all() or (frame<=0).any().any():
            raise ValueError('Invalid retained prices. OOS missing observations require a delisting/missing-data policy; do not drop assets using future availability.')
    return train,test,report
