import pandas as pd

from fire_es.cleaning import build_event_identity


def test_event_identity_groups_cross_sheet_duplicates():
    df = pd.DataFrame(
        [
            {
                "row_id": 101,
                "source_sheet": "БД-1 ... 2000-2020 (1+2)",
                "fire_date": "2010-01-02",
                "year": 2010,
                "region_code": 77,
                "settlement_type_code": 1,
                "enterprise_type_code": 11,
                "building_floors": 9,
                "fire_floor": 3,
                "source_item_code": 12,
                "object_name": "Склад",
                "address": "Улица 1",
                "equipment_count": 4,
            },
            {
                "row_id": 101,
                "source_sheet": "1...2000-2008",
                "fire_date": "2010-01-02",
                "year": 2010,
                "region_code": 77,
                "settlement_type_code": 1,
                "enterprise_type_code": 11,
                "building_floors": 9,
                "fire_floor": 3,
                "source_item_code": 44,
                "object_name": "Склад",
                "address": "Улица 1",
                "equipment_count": 3,
            },
        ]
    )

    result = build_event_identity(df)
    assert result["duplicate_group_id"].nunique() == 1
    assert result["is_canonical_event_record"].sum() == 1
    assert set(result["duplicate_policy"]) == {"conflict_kept"}
