import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
sys.path.insert(0, str(SCRIPTS))

from train_and_explain import load_data, build_preprocessor


def test_load_data_and_preprocessor():
    data_path = ROOT / 'data' / 'GlobalTechTalent_50k.csv'
    df = load_data(data_path)
    assert 'Migration_Status' in df.columns
    assert 'Visa_Approval_Date' not in df.columns
    preprocessor, categorical, numeric = build_preprocessor(df)
    # ensure preprocessor returns transformers
    assert hasattr(preprocessor, 'transform')
    assert len(categorical) > 0
    assert len(numeric) > 0
