from fire_es.rank_tz_contract import get_feature_set_spec


def test_dispatch_feature_set_excludes_future_features():
    spec = get_feature_set_spec("dispatch_initial_safe")
    forbidden = {
        "source_item_code",
        "t_arrival_min",
        "t_first_hose_min",
        "t_contained_min",
        "t_extinguished_min",
        "fatalities",
        "injuries",
        "direct_damage",
        "equipment_count",
        "nozzle_count",
    }
    assert not forbidden.intersection(spec["feature_order"])
