import pandas as pd

from fire_es.equipment_parse import analyze_equipment_parse


def test_analyze_equipment_parse_missing_equipment():
    analysis = analyze_equipment_parse(None, declared_count=None)
    assert analysis["resource_parse_confidence"] == 0.0
    assert "missing_or_unparsed_resources" in analysis["resource_parse_flags"]
    assert analysis["resource_count_parsed"] == 0


def test_analyze_equipment_parse_unknown_code():
    analysis = analyze_equipment_parse("999", declared_count=1)
    assert analysis["resource_count_parsed"] == 0
    assert "missing_or_unparsed_resources" in analysis["resource_parse_flags"]


def test_analyze_equipment_parse_count_conflict():
    analysis = analyze_equipment_parse("11, 23", declared_count=5)
    assert analysis["resource_count_parsed"] == 2
    assert analysis["resource_count_conflict"] == 3.0
    assert "resource_parse_conflict" in analysis["resource_parse_flags"]
