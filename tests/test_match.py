import exchange_calendars as xc
import pandas as pd

from insider_screen.match import match, normalize


def test_normalize():
    assert normalize("APPLE INC /CA/") == "apple"
    assert normalize("Kindred Biosciences, Inc.") == "kindred biosciences"
    assert normalize("Johnson & Johnson") == "johnson and johnson"
    assert normalize("The Goodness Growth Holdings, Inc.") == "goodness growth"


CAL = xc.get_calendar("XNYS", start="2015-01-01")
COMPANIES = pd.DataFrame({
    "cik": [1, 2, 3],
    "name": ["KINDRED BIOSCIENCES, INC.", "NEOPHOTONICS CORP", "UNRELATED CO"],
    "former_names": ["", "", ""],
})
EVENTS = pd.DataFrame({
    "event_id": ["acq-1", "ear-1", "acq-2"],
    "cik": [1, 1, 2],
    "event_type": ["acquisition_target", "earnings", "acquisition_target"],
    "day0": pd.to_datetime(["2021-06-16", "2021-06-17", "2021-11-04"]),
})


def traded(name, ann, etype="acquisition_target"):
    return pd.DataFrame({"lr_no": [1], "issuer_name": [name], "announcement_date": [pd.Timestamp(ann) if ann else pd.NaT],
                         "event_type": [etype]})


def test_exact_and_type_preference():
    out = match(traded("Kindred Biosciences, Inc.", "2021-06-16"), COMPANIES, EVENTS, CAL).iloc[0]
    assert (out.cik, out.name_method, out.event_id, out.session_gap) == (1, "exact", "acq-1", 0)


def test_fuzzy_name_and_gap():
    out = match(traded("NeoPhotonics Corporation Inc", "2021-11-03"), COMPANIES, EVENTS, CAL).iloc[0]
    assert out.event_id == "acq-2" and out.session_gap == 1


def test_reasons():
    assert match(traded("Zzyzx Widgets", "2021-06-16"), COMPANIES, EVENTS, CAL).iloc[0].reason == "no_company"
    assert match(traded("Kindred Biosciences", None), COMPANIES, EVENTS, CAL).iloc[0].reason == "no_announcement_date"
    assert match(traded("Kindred Biosciences", "2021-03-01"), COMPANIES, EVENTS, CAL).iloc[0].reason == "no_event_in_window"
