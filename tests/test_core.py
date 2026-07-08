import pandas as pd
from io import StringIO
from src.pyfolio.core import compute_correlation

def test_compute_correlation():
    csv = StringIO("""Date,A,B,C
2020-01-01,1.0,2.0,3.0
2020-01-02,2.0,4.0,6.0
2020-01-03,3.0,6.0,9.0
""")
    df = pd.read_csv(csv)
    corr = compute_correlation(df)
    # Perfect correlation between columns that are multiples
    assert corr.shape == (3, 3)
    assert round(corr.loc["A", "B"], 5) == 1.0
