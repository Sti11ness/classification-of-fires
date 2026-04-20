from fire_es.rank_tz_contract import get_feature_set_spec


def test_dispatch_feature_set_excludes_future_features():
    spec = get_feature_set_spec("dispatch_initial_safe")
    forbidden = {"t_arrival_min", "t_first_hose_min", "fatalities", "injuries", "direct_damage", "equipment_count"}
    assert not forbidden.intersection(spec["feature_order"])
