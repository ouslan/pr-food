import duckdb


def get_conn(db_path: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path)


def init_zipstable(ddb_path: str) -> None:
    conn = get_conn(ddb_path)
    conn.load_extension("spatial")
    conn.sql(
        """
        CREATE TABLE IF NOT EXISTS "zipstable" (
            zipcode VARCHAR(5) NOT NULL,
            geom GEOMETRY
        );
    """
    )
