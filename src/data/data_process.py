import logging
import os

import altair as alt
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import requests
from pysal.lib import weights
from shapely import wkt

from ..jp_qcew.src.data.data_process import cleanData
from ..models import init_death_table, init_dp03_table


class FoodDeseart(cleanData):
    def __init__(
        self,
        saving_dir: str = "data/",
        database_file: str = "data.ddb",
        log_file: str = "data_process.log",
    ):
        super().__init__(saving_dir, database_file, log_file)

    def food_data(self) -> gpd.GeoDataFrame:
        if "qcewtable" not in self.conn.sql("SHOW TABLES;").df().get("name").tolist():
            self.make_qcew_dataset()
        df = self.conn.sql(
            f"""
            SELECT year,qtr,phys_addr_5_zip,naics_code,ein FROM 'qcewtable';
             """
        ).pl()
        df = df.filter(pl.col("phys_addr_5_zip") != "")
        df = df.with_columns(
            pl.col("phys_addr_5_zip").cast(pl.String).str.zfill(5).alias("zipcode"),
            pl.when(pl.col("naics_code").cast(pl.String).str.starts_with("4451"))
            .then(1)
            .otherwise(0)
            .alias("supermarkets_and_others"),
            pl.when(pl.col("naics_code").cast(pl.String).str.starts_with("44511"))
            .then(1)
            .otherwise(0)
            .alias("supermarkets"),
            pl.when(pl.col("naics_code").cast(pl.String).str.starts_with("44513"))
            .then(1)
            .otherwise(0)
            .alias("convenience_retailers"),
            pl.when(pl.col("naics_code").cast(pl.String).str.starts_with("4452"))
            .then(1)
            .otherwise(0)
            .alias("whole_foods"),
            pl.when(pl.col("naics_code").cast(pl.String).str.starts_with("23"))
            .then(1)
            .otherwise(0)
            .alias("construction"),
            pl.when(pl.col("naics_code").cast(pl.String).str.starts_with("52"))
            .then(1)
            .otherwise(0)
            .alias("finance"),
            pl.when(pl.col("ein").cast(pl.String).str.starts_with("911223280"))
            .then(1)
            .otherwise(0)
            .alias("costco"),
            pl.when(pl.col("ein").cast(pl.String).str.starts_with("660475164"))
            .then(1)
            .otherwise(0)
            .alias("walmart"),
        )

        df = df.group_by(["year", "qtr", "zipcode"]).agg(
            supermarkets_and_others=pl.col("supermarkets_and_others").sum(),
            supermarkets=pl.col("supermarkets").sum(),
            convenience_retailers=pl.col("convenience_retailers").sum(),
            whole_foods=pl.col("whole_foods").sum(),
            construction=pl.col("construction").sum(),
            finance=pl.col("finance").sum(),
        )

        df = df.with_columns(
            total_food=pl.col("supermarkets") + pl.col("convenience_retailers")
        )
        gdf = self.spatial_data()

        gdf = gdf.join(
            df.to_pandas().set_index("zipcode"),
            on="zipcode",
            how="inner",
            validate="1:m",
        )

        gdf = gdf.reset_index(drop=True)
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry")
        gdf["supermarkets_and_others_area"] = gdf["supermarkets_and_others"] / (
            gdf.area * 1000
        )
        gdf["supermarkets_area"] = gdf["supermarkets"] / (gdf.area * 1000)
        gdf["convenience_retailers_area"] = gdf["convenience_retailers"] / (
            gdf.area * 1000
        )
        gdf["whole_foods_area"] = gdf["whole_foods"] / (gdf.area * 1000)
        gdf["total_food_area"] = gdf["total_food"] / (gdf.area * 1000)
        gdf["construction_area"] = gdf["construction"] / (gdf.area * 1000)
        gdf["finance_area"] = gdf["finance"] / (gdf.area * 1000)

        return gdf

    def process_death(self) -> pl.DataFrame:
        df = self.pull_death()
        df = df.with_columns(
            qtr=pl.when(
                (pl.col("deathdate_month") > 0) & (pl.col("deathdate_month") <= 3)
            )
            .then(1)
            .when((pl.col("deathdate_month") > 3) & (pl.col("deathdate_month") <= 6))
            .then(2)
            .when((pl.col("deathdate_month") > 6) & (pl.col("deathdate_month") <= 9))
            .then(3)
            .when((pl.col("deathdate_month") > 9) & (pl.col("deathdate_month") <= 12))
            .then(4)
            .otherwise(None)
        ).drop("deathdate_month")
        df = df.rename({"deathdate_year": "year"})
        df = df.group_by("year", "qtr", "zipcode").agg(
            pl.col("death_a").sum().alias("paracites_disease"),
            pl.col("death_c").sum().alias("cancer_disease"),
            pl.col("death_g").sum().alias("nervous_disease"),
            pl.col("death_j").sum().alias("respiratory_disease"),
            pl.col("death_i").sum().alias("circulatory_disease"),
        )
        return df

    def make_dataset(self):
        death_df = self.process_death()
        gdf = self.food_data()

        dp03_df = self.pull_dp03()
        dp03_df = dp03_df.with_columns(qtr=4)
        gdf = gdf.merge(
            dp03_df.to_pandas(),
            on=["year", "qtr", "zipcode"],
            how="inner",
            validate="1:1",
        )
        gdf = gdf.sort_values(by=["zipcode", "year", "qtr"]).reset_index(drop=True)
        columns = [
            "total_population",
            "inc_25k_35k",
            "inc_35k_50k",
            "inc_50k_75k",
            "inc_75k_100k",
            "inc_100k_150k",
            "inc_150k_200k",
            "inc_more_200k",
        ]
        for col in columns:
            gdf[col] = gdf.groupby("zipcode")[col].transform(
                lambda group: group.interpolate(method="cubic")
            )
        gdf = gdf.merge(
            death_df.to_pandas(),
            on=["year", "qtr", "zipcode"],
            how="inner",
            validate="1:1",
        )

        gdf["paracites_by_pop"] = gdf["paracites_disease"] / gdf["total_population"]
        gdf["cancer_by_pop"] = gdf["cancer_disease"] / gdf["total_population"]
        gdf["nervous_by_pop"] = gdf["nervous_disease"] / gdf["total_population"]
        gdf["respiratory_by_pop"] = gdf["respiratory_disease"] / gdf["total_population"]
        gdf["circulatory_by_pop"] = gdf["circulatory_disease"] / gdf["total_population"]

        gdf = gdf.sort_values(by=["year", "qtr", "zipcode"]).reset_index(drop=True)
        w = weights.Queen.from_dataframe(gdf[(gdf["year"] == 2017) & (gdf["qtr"] == 1)])
        spatial_lag_results = []
        for year in range(2015, 2020):
            for qtr in range(1, 5):
                group_df = gdf[(gdf["year"] == year) & (gdf["qtr"] == qtr)].reset_index(
                    drop=True
                )
                spatial_paracites = self.calculate_spatial_lag(
                    group_df, w, "paracites_by_pop"
                )
                spatial_cancer = self.calculate_spatial_lag(
                    group_df, w, "cancer_by_pop"
                )
                spatial_nervouse = self.calculate_spatial_lag(
                    group_df, w, "nervous_by_pop"
                )
                spatial_respiratory = self.calculate_spatial_lag(
                    group_df, w, "respiratory_by_pop"
                )
                spatial_circulatory = self.calculate_spatial_lag(
                    group_df, w, "circulatory_by_pop"
                )
                spatial_supermarkets_and_others_area = self.calculate_spatial_lag(
                    group_df, w, "supermarkets_and_others_area"
                )
                spatial_supermarkets_area = self.calculate_spatial_lag(
                    group_df, w, "supermarkets_area"
                )
                spatial_convenience_retailers_area = self.calculate_spatial_lag(
                    group_df, w, "convenience_retailers_area"
                )
                spatial_whole_foods_area = self.calculate_spatial_lag(
                    group_df, w, "whole_foods_area"
                )
                spatial_total_food_area = self.calculate_spatial_lag(
                    group_df, w, "total_food_area"
                )
                spatial_construction_area = self.calculate_spatial_lag(
                    group_df, w, "construction_area"
                )
                spatial_finance_area = self.calculate_spatial_lag(
                    group_df, w, "finance_area"
                )
                # Add the spatial lag results back to the group dataframe
                group_df["w_paracites"] = spatial_paracites.flatten()
                group_df["w_cancer"] = spatial_cancer.flatten()
                group_df["w_nervouse"] = spatial_nervouse.flatten()
                group_df["w_respiratory"] = spatial_respiratory.flatten()
                group_df["w_circulatory"] = spatial_circulatory.flatten()
                group_df["w_supermarkets_and_others_area"] = (
                    spatial_supermarkets_and_others_area.flatten()
                )
                group_df["w_supermarkets_area"] = spatial_supermarkets_area.flatten()
                group_df["w_convenience_retailers_area"] = (
                    spatial_convenience_retailers_area.flatten()
                )
                group_df["w_whole_foods_area"] = spatial_whole_foods_area.flatten()
                group_df["w_total_food_area"] = spatial_total_food_area.flatten()
                group_df["w_construction_area"] = spatial_construction_area.flatten()
                group_df["w_finance_area"] = spatial_finance_area.flatten()

                # Append the group to the results list
                spatial_lag_results.append(group_df)
        gdf = pd.concat(spatial_lag_results)

        return gdf[(gdf["year"] >= 2015) & (gdf["year"] <= 2019)]

    def pull_death(self) -> pl.DataFrame:
        if "DeathTable" not in self.conn.sql("SHOW TABLES;").df().get("name").tolist():
            init_death_table(self.data_file)
            df = pd.read_stata(f"{self.saving_dir}external/deaths.dta")
            df = df[~df["zipcode"].isna()].reset_index(drop=True)
            df = df[df["deathdate_month"] <= 12].reset_index(drop=True)
            df["zipcode"] = df["zipcode"].astype(int).astype(str).str.zfill(5)
            self.conn.sql("INSERT INTO 'DeathTable' BY NAME SELECT * FROM df")
            logging.info(f"succesfully inserting DeathTable")
            return self.conn.sql("SELECT * FROM 'DeathTable';").pl()
        else:
            return self.conn.sql("SELECT * FROM 'DeathTable';").pl()

    def pull_query(self, params: list, year: int) -> pl.DataFrame:
        # prepare custom census query
        param = ",".join(params)
        base = "https://api.census.gov/data/"
        flow = "/acs/acs5/profile"
        url = f"{base}{year}{flow}?get={param}&for=zip%20code%20tabulation%20area:*&in=state:72"
        logging.debug(url)
        df = pl.DataFrame(requests.get(url).json())

        # get names from DataFrame
        names = df.select(pl.col("column_0")).transpose()
        names = names.to_dicts().pop()
        names = dict((k, v.lower()) for k, v in names.items())

        # Pivot table
        df = df.drop("column_0").transpose()
        return df.rename(names).with_columns(year=pl.lit(year))

    def make_spatial_table(self):
        # initiiate the database tables
        if "zipstable" not in self.conn.sql("SHOW TABLES;").df().get("name").tolist():
            # Download the shape files
            if not os.path.exists(f"{self.saving_dir}external/zips_shape.zip"):
                self.pull_file(
                    url="https://www2.census.gov/geo/tiger/TIGER2024/ZCTA520/tl_2024_us_zcta520.zip",
                    filename=f"{self.saving_dir}external/zips_shape.zip",
                )
                logging.info("Downloaded zipcode shape files")

            # Process and insert the shape files
            gdf = gpd.read_file(f"{self.saving_dir}external/zips_shape.zip")
            gdf = gdf[gdf["ZCTA5CE20"].str.startswith("00")]
            gdf = gdf.rename(columns={"ZCTA5CE20": "zipcode"}).reset_index()
            gdf = gdf[["zipcode", "geometry"]]
            gdf["zipcode"] = gdf["zipcode"].str.strip()
            df = gdf.drop(columns="geometry")
            geometry = gdf["geometry"].apply(lambda geom: geom.wkt)
            df["geometry"] = geometry
            self.conn.execute("CREATE TABLE zipstable AS SELECT * FROM df")
            logging.info(
                f"The zipstable is empty inserting {self.saving_dir}external/cousub.zip"
            )
            return self.conn.sql("SELECT * FROM zipstable;")
        else:
            return self.conn.sql("SELECT * FROM zipstable;")

    def spatial_data(self) -> gpd.GeoDataFrame:
        gdf = gpd.GeoDataFrame(self.make_spatial_table().df())
        gdf["geometry"] = gdf["geometry"].apply(wkt.loads)
        gdf = gdf.set_geometry("geometry").set_crs("EPSG:4269", allow_override=True)
        gdf = gdf.to_crs("EPSG:3395")
        gdf["zipcode"] = gdf["zipcode"].astype(str)
        return gdf

    def pull_dp03(self) -> pl.DataFrame:
        if "DP03Table" not in self.conn.sql("SHOW TABLES;").df().get("name").tolist():
            init_dp03_table(self.data_file)
        for _year in range(2012, 2020):
            if (
                self.conn.sql(f"SELECT * FROM 'DP03Table' WHERE year={_year}")
                .df()
                .empty
            ):
                try:
                    logging.info(f"pulling {_year} data")
                    df = self.pull_query(
                        params=[
                            "DP03_0001E",
                            "DP03_0003E",
                            "DP03_0004E",
                            "DP03_0005E",
                            "DP03_0006E",
                            "DP03_0007E",
                            "DP03_0014E",
                            "DP03_0025E",
                            "DP03_0033E",
                            "DP03_0051E",
                            "DP03_0052E",
                            "DP03_0053E",
                            "DP03_0054E",
                            "DP03_0055E",
                            "DP03_0056E",
                            "DP03_0057E",
                            "DP03_0058E",
                            "DP03_0059E",
                            "DP03_0060E",
                            "DP03_0061E",
                        ],
                        year=_year,
                    )
                    df = df.rename(
                        {
                            "dp03_0001e": "total_population",
                            "dp03_0003e": "total_civilian_force",
                            "dp03_0004e": "total_labor_force",
                            "dp03_0005e": "total_unemployed",
                            "dp03_0006e": "total_armed_forces",
                            "dp03_0007e": "total_not_labor",
                            "dp03_0014e": "total_own_children",
                            "dp03_0025e": "mean_travel_time",
                            "dp03_0033e": "agr_fish_employment",
                            "dp03_0051e": "total_house",
                            "dp03_0052e": "inc_less_10k",
                            "dp03_0053e": "inc_10k_15k",
                            "dp03_0054e": "inc_15k_25k",
                            "dp03_0055e": "inc_25k_35k",
                            "dp03_0056e": "inc_35k_50k",
                            "dp03_0057e": "inc_50k_75k",
                            "dp03_0058e": "inc_75k_100k",
                            "dp03_0059e": "inc_100k_150k",
                            "dp03_0060e": "inc_150k_200k",
                            "dp03_0061e": "inc_more_200k",
                        }
                    )
                    df = df.rename({"zip code tabulation area": "zipcode"}).drop(
                        ["state"]
                    )
                    df = df.with_columns(
                        pl.all().exclude("zipcode", "mean_travel_time").cast(pl.Int64)
                    )
                    self.conn.sql("INSERT INTO 'DP03Table' BY NAME SELECT * FROM df")
                    logging.info(f"succesfully inserting {_year}")
                except:
                    logging.warning(f"The ACS for {_year} is not availabe")
                    continue
            else:
                logging.info(f"data for {_year} is in the database")
                continue
        return self.conn.sql("SELECT * FROM 'DP03Table';").pl()

    def gen_food_graph(self, var: str, year: int, qtr: int, title: str):
        # define data
        df = self.food_data(year=year, qtr=qtr)

        # define choropleth scale
        quant = df[var]
        domain = [
            0,
            quant.quantile(0.25),
            quant.quantile(0.50),
            quant.quantile(0.75),
            quant.max(),
        ]
        scale = alt.Scale(domain=domain, scheme="viridis")
        # define choropleth chart
        choropleth = (
            alt.Chart(df, title=title)
            .mark_geoshape()
            .transform_lookup(
                lookup="zipcode",
                from_=alt.LookupData(data=df, key="zipcode", fields=[var]),
            )
            .encode(
                alt.Color(
                    f"{var}:Q",
                    scale=scale,
                    legend=alt.Legend(direction="horizontal", orient="bottom"),
                )
            )
            .project(type="mercator")
            .properties(width="container", height=300)
        )
        return choropleth

    def calculate_spatial_lag(self, df, w, column) -> np.ndarray:
        # Reshape y to match the number of rows in the dataframe
        y = df[column].values.reshape(-1, 1)

        # Apply spatial lag
        spatial_lag = weights.lag_spatial(w, y)

        return spatial_lag
