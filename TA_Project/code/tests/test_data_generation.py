import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
sys.path.insert(0, str(SCRIPTS))

from generate_synthetic_data import generate_dataset


def test_generate_dataset_shape_and_columns():
    df = generate_dataset(n_rows=1000, seed=0)
    assert df.shape[0] == 1000
    expected = {'UserID','Country_Origin','GitHub_Activity','Research_Citations','Migration_Status'}
    assert expected.issubset(set(df.columns))
    assert set(df["Migration_Status"].unique()).issubset({0, 1})
