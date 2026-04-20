import pandas as pd

from fire_es.ranking import assign_rank_tz


def test_vector_rank_missing_or_unparsed_resources_are_not_trainable():
    df = pd.DataFrame(
        {
            "equipment": [None, "999", "11, 23", "11, 23"],
            "equipment_count": [None, 1, 5, 2],
            "is_canonical_event_record": [True, True, True, True],
        }
    )
    result = assign_rank_tz(df, target_definition="vector")
    assert pd.isna(result.loc[0, "rank_tz_vector"])
    assert pd.isna(result.loc[1, "rank_tz_vector"])
    assert "missing_or_unparsed_resources" in result.loc[0, "rank_quality_flags"]
    assert "missing_or_unparsed_resources" in result.loc[1, "rank_quality_flags"]
    assert result.loc[2, "usable_for_training"] == 0
    assert "resource_parse_conflict" in result.loc[2, "rank_quality_flags"]
    assert result.loc[3, "rank_tz_vector"] == 2.0
    assert result.loc[3, "usable_for_training"] == 1
