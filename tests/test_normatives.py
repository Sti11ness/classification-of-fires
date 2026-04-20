from fire_es.db import init_db
from fire_es.normatives import get_normative_rank_table, load_rank_resource_normatives
from sqlalchemy import text


def test_normatives_json_and_db_share_same_source(tmp_path):
    db = init_db(str(tmp_path / "normatives.sqlite"))
    session = db.get_session()
    try:
        rows = session.execute(
            text("SELECT rank, resource_type, quantity, normative_version FROM normatives")
        ).fetchall()
        payload = load_rank_resource_normatives()
        rank_table = get_normative_rank_table(payload)
        assert len(rows) > 0
        assert {row[3] for row in rows} == {payload["normative_version"]}
        assert set(rank_table["rank"].tolist()) == {1.0, 1.5, 2.0, 3.0, 4.0, 5.0}
    finally:
        session.close()
        db.close()
