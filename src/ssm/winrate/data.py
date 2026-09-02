"""Panel assembly for the winrate surfaces.

Builds the OHLCV + DXY + on-chain frame the nodes are computed over. OHLCV and
DXY already exist in the stage-1 ODS; the CoinMetrics on-chain metrics do not --
``step_1_data_ingestion`` deliberately drops everything but ``PriceUSD``. Rather
than change that stage's artifact contract, the on-chain columns are fetched once
and cached alongside it, so every downstream run is offline and deterministic.
"""

import json
import urllib.parse
import urllib.request

import pandas as pd

from ssm.config import OUTPUT, ODS_CSV, ensure_dirs

ONCHAIN_CSV = OUTPUT / 'onchain_cache.csv'

_CM_METRICS = {
    'CapMVRVCur': 'mvrv',
    'HashRate':   'hash_rate',
    'AdrActCnt':  'adr_act_cnt',
    'TxCnt':      'tx_cnt',
}


def download_onchain() -> pd.DataFrame:
    """Fetch the CoinMetrics community on-chain metrics used by the on_chain family."""
    params = {
        'assets':    'btc',
        'metrics':   ','.join(_CM_METRICS),
        'frequency': '1d',
        'format':    'json',
        'page_size': '10000',
    }
    url = ('https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?'
           + urllib.parse.urlencode(params))
    with urllib.request.urlopen(url, timeout=60) as r:
        payload = json.load(r)

    df = pd.DataFrame(payload['data'])
    df['Date'] = pd.to_datetime(df['time'], utc=True).dt.tz_localize(None)
    df = df.set_index('Date').sort_index()
    return df[list(_CM_METRICS)].rename(columns=_CM_METRICS).apply(pd.to_numeric, errors='coerce')


def load_onchain(refresh: bool = False) -> pd.DataFrame:
    """On-chain metrics from the local cache, fetching once if absent or if forced."""
    ensure_dirs()
    if refresh or not ONCHAIN_CSV.exists():
        print('Fetching CoinMetrics on-chain metrics...')
        df = download_onchain()
        df.to_csv(ONCHAIN_CSV)
        print(f'Cached -> {ONCHAIN_CSV}  ({len(df)} rows, {df.index[0].date()} -> {df.index[-1].date()})')
    return pd.read_csv(ONCHAIN_CSV, index_col='Date', parse_dates=True)


def load_panel(start: str = '2015-01-01', refresh_onchain: bool = False) -> pd.DataFrame:
    """OHLCV + DXY + on-chain on one daily spine, dropping rows without a close.

    ``start`` matches the winrate-matrix BTC workspace so the surfaces are computed
    over the same history the sibling project reports on.
    """
    ods = pd.read_csv(ODS_CSV, index_col='Date', parse_dates=True)
    ods = ods[['open', 'high', 'low', 'close', 'volume', 'dxy']]

    panel = ods.join(load_onchain(refresh_onchain), how='left')
    panel = panel[panel.index >= pd.Timestamp(start)]
    panel = panel.dropna(subset=['close'])
    panel['dxy'] = panel['dxy'].ffill()
    return panel
