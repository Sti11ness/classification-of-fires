from fire_es.rank_tz_contract import (
    AVAILABILITY_STAGE_ARRIVAL,
    AVAILABILITY_STAGE_FIRST_HOSE,
    validate_stage_input_requirements,
)


def test_arrival_stage_requires_arrival_time():
    errors = validate_stage_input_requirements(
        {
            "region_code": 77,
            "t_report_min": 15,
        },
        availability_stage=AVAILABILITY_STAGE_ARRIVAL,
    )
    assert errors == ["t_arrival_min"]


def test_first_hose_stage_requires_hose_time():
    errors = validate_stage_input_requirements(
        {
            "region_code": 77,
            "t_report_min": 15,
            "t_arrival_min": 22,
        },
        availability_stage=AVAILABILITY_STAGE_FIRST_HOSE,
    )
    assert errors == ["t_first_hose_min"]
