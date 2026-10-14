"""Backwards-compatible cleaning helper."""
from Prep_Data import Prepare_Data

def Clean(df):
    return Prepare_Data(df, show_details=False)
