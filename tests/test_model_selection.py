import pandas as pd

from fire_es.model_selection import split_dataset


def test_group_shuffle_has_zero_event_overlap():
    df = pd.DataFrame(
        {
            "event_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "fire_date": pd.date_range("2025-01-01", periods=8, freq="D"),
            "rank_tz_vector": [1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 3.0, 3.0],
        }
    )
    y = pd.Series([1, 1, 2, 2, 3, 3, 4, 4])
    split = split_dataset(df, y=y, split_protocol="group_shuffle", test_size=0.25)
    assert split.metadata["event_overlap_rate"] == 0.0
    train_events = set(df.iloc[split.train_indices]["event_id"])
    test_events = set(df.iloc[split.test_indices]["event_id"])
    assert train_events.isdisjoint(test_events)
