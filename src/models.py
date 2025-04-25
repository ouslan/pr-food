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


def init_dp03_table(db_path: str) -> None:
    conn = get_conn(db_path=db_path)

    conn.sql(
        """
        CREATE TABLE IF NOT EXISTS "DP03Table" (
            year INTEGER,
            zipcode VARCHAR(5),
            total_population INTEGER,
            total_civilian_force INTEGER,
            total_labor_force INTEGER,
            total_unemployed INTEGER,
            total_armed_forces INTEGER,
            total_not_labor INTEGER,
            total_own_children INTEGER,
            mean_travel_time FLOAT,
            agr_fish_employment INTEGER,
            total_house INTEGER,
            inc_less_10k INTEGER,
            inc_10k_15k INTEGER,
            inc_15k_25k INTEGER,
            inc_25k_35k INTEGER,
            inc_35k_50k INTEGER,
            inc_50k_75k INTEGER,
            inc_75k_100k INTEGER,
            inc_100k_150k INTEGER,
            inc_150k_200k INTEGER,
            inc_more_200k INTEGER
        );
        """
    )


def init_death_table(db_path: str) -> None:
    conn = get_conn(db_path=db_path)

    conn.sql(
        """
        CREATE TABLE IF NOT EXISTS DeathTable (
            deathdate_year INTEGER,
            deathdate_month INTEGER,
            zipcode VARCHAR(5),
            death_a INTEGER,
            death_b INTEGER,
            death_c INTEGER,
            death_d INTEGER,
            death_e INTEGER,
            death_f INTEGER,
            death_g INTEGER,
            death_h INTEGER,
            death_i INTEGER,
            death_j INTEGER,
            death_k INTEGER,
            death_l INTEGER,
            death_m INTEGER,
            death_n INTEGER,
            death_o INTEGER,
            death_p INTEGER,
            death_q INTEGER,
            death_r INTEGER,
            death_s INTEGER,
            death_t INTEGER,
            death_u INTEGER,
            death_v INTEGER,
            death_w INTEGER,
            muertes INTEGER
        );
        """
    )
