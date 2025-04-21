import logging
import os
from datetime import datetime

import altair as alt
import geopandas as gpd
import polars as pl
import requests
from ..jp_qcew.src.data.data_process import cleanData
from ..models import init_dp03_table


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
        gdf = self.make_spatial_table()

        gdf = gdf.join(
            df.to_pandas().set_index("zipcode"),
            on="zipcode",
            how="inner",
            validate="1:1",
        )
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

    def pull_query(self, params: list, year: int) -> pl.DataFrame:
        # prepare custom census query
        param = ",".join(params)
        base = "https://api.census.gov/data/"
        flow = "/acs/acs5/profile"
        url = f"{base}{year}{flow}?get={param}&for=zip%20code%20tabulation%20area:*&in=state:72"
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

    def pull_dp03(self) -> pl.DataFrame:
        if "DP03Table" not in self.conn.sql("SHOW TABLES;").df().get("name").tolist():
            init_dp03_table(self.data_file)
        for _year in range(2012, datetime.now().year):
            if (
                self.conn.sql(f"SELECT * FROM 'DP03Table' WHERE year={_year}")
                .df()
                .empty
            ):
                try:
                    logging.info(f"pulling {_year} data")
                    tmp = self.pull_query(
                        params=[
                            "DP03_0001E",
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
                    tmp = tmp.rename(
                        {
                            "dp03_0001e": "total_population",
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
                    tmp = tmp.rename({"zip code tabulation area": "zipcode"}).drop(
                        ["state"]
                    )
                    tmp = tmp.with_columns(pl.all().exclude("zipcode").cast(pl.Int64))
                    self.conn.sql("INSERT INTO 'DP03Table' BY NAME SELECT * FROM tmp")
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
