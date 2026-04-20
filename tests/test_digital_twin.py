import pandas as pd

from fire_es.simulation.digital_twin import build_statistical_profile, generate_synthetic_fire_dataset


def test_digital_twin_generates_synthetic_dataset():
    df = pd.DataFrame(
        {
            "region_code": [77, 78, 79],
            "distance_to_station": [1.0, 2.0, 3.0],
            "rank_tz_vector": [1.0, 1.5, 2.0],
        }
    )
    synthetic, profile = generate_synthetic_fire_dataset(df, n_rows=10, missing_rate=0.1, noise_scale=0.1)
    assert len(synthetic) == 10
    assert profile["row_count"] == 10
    assert "missingness" in build_statistical_profile(df)
