import polars as pl
import requests
import os
import logging
import geopandas as gpd
from shapely import wkt
from ..jp_qcew.src.data.data_process import cleanData


class foodDeseart(cleanData):
    def __init__(
        self,
        saving_dir: str = "data/",
        database_file: str = "data.ddb",
        log_file: str = "data_process.log",
    ):
        super().__init__(saving_dir, database_file, log_file)

    def buiss_data(self, year: int, qtr: int) -> gpd.GeoDataFrame:
        if "qcewtable" not in self.conn.sql("SHOW TABLES;").df().get("name").tolist():
            self.make_qcew_dataset()
        df = self.conn.sql(
            f"""
            SELECT year,qtr,phys_addr_5_zip,naics_code FROM 'qcewtable' 
                WHERE year = {year} AND qtr = {qtr};
             """
        ).pl()

        df = df.with_columns(
            zipcode=pl.col("phys_addr_5_zip").cast(pl.String).str.zfill(5),
            naics4=pl.col("naics_code").str.slice(0, 4),
            dummy=pl.lit(1),
        )
        df = df.filter(pl.col("phys_addr_5_zip") != "")
        df = df.bottom_kwith_columns(
                                        pl.col("phys_addr_5_zip").cast(pl.String).str.zfill(5).alias("zipcode"),
                                      ((pl.col("first_month_employment") + pl.col("second_month_employment") + pl.col("third_month_employment"))/3).alias("total_employment"),
                                        pl.when(pl.col("naics_code").cast(pl.String).str.starts_with("4451")).then(1).otherwise(0).alias("supermarkets_and_others"),
                                        pl.when(pl.col("naics_code").cast(pl.String).str.starts_with("44511")).then(1).otherwise(0).alias("supermarkets"),
                                        pl.when(pl.col("naics_code").cast(pl.String).str.starts_with("44513")).then(1).otherwise(0).alias("convenience_retailers"),
                                        pl.when(pl.col("naics_code").cast(pl.String).str.starts_with("4452")).then(1).otherwise(0).alias("whole_foods"),
                                        pl.when(pl.col("ein").cast(pl.String).str.starts_with("911223280")).then(1).otherwise(0).alias("costco"),
                                        pl.when(pl.col("ein").cast(pl.String).str.starts_with("660475164")).then(1).otherwise(0).alias("walmart"))


        df = df.group_by(["year", "qtr", "naics4", "zipcode"]).agg(
            buisnesses=pl.col("dummy").sum()
        )
        gdf = self.make_spatial_table()

        gdf = gdf.join(
            df.to_pandas().set_index("zipcode"),
            on="zipcode",
            how="inner",
            validate="1:m",
        )
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry")

        gdf["buis_area"] = gdf["buisnesses"] / gdf.area

        return gdf

    def pull_query(self, params: list, year: int) -> pl.DataFrame:
        # prepare custom census query
        param = ",".join(params)
        base = "https://api.census.gov/data/"
        flow = "/acs/acs5/profile"
        url = f"{base}{year}{flow}?get={param}&for=county%20subdivision:*&in=state:72&in=county:*"
        df = pl.DataFrame(requests.get(url).json())

        # get names from DataFrame
        names = df.select(pl.col("column_0")).transpose()
        names = names.to_dicts().pop()
        names = dict((k, v.lower()) for k, v in names.items())

        # Pivot table
        df = df.drop("column_0").transpose()
        return df.rename(names).with_columns(year=pl.lit(year))

    def make_spatial_table(self) -> gpd.GeoDataFrame:
        # pull shape files from the census
        if not os.path.exists(f"{self.saving_dir}external/zips_shape.zip"):
            self.pull_file(
                url="https://www2.census.gov/geo/tiger/TIGER2024/ZCTA520/tl_2024_us_zcta520.zip",
                filename=f"{self.saving_dir}external/zips_shape.zip",
            )
            logging.info("Downloaded zipcode shape files")
        # initiiate the database tables
        if "zipstable" not in self.conn.sql("SHOW TABLES;").df().get("name").tolist():
            gdf = gpd.read_file(f"{self.saving_dir}external/zips_shape.zip")
            gdf = gdf.to_crs("EPSG:4326")
            gdf = gdf.rename(columns={"ZCTA5CE20": "zipcode"})
            gdf = gdf[gdf["zipcode"].astype(int) < 999].reset_index()
            gdf = gdf[["zipcode", "geometry"]]
            df = gdf.drop(columns="geometry")
            geometry = gdf["geometry"].apply(lambda geom: geom.wkt)
            df["geometry"] = geometry
            self.conn.execute("CREATE TABLE zipstable AS SELECT * FROM df")
            logging.info("Inserted zipcode shape files into database")

            gdf = self.conn.sql("SELECT * FROM zipstable;").df()
            gdf["geometry"] = gdf["geometry"].apply(wkt.loads)
            gdf = gdf.set_geometry("geometry")
            return gdf
        else:
            gdf = self.conn.sql("SELECT * FROM zipstable;").df()
            gdf["geometry"] = gdf["geometry"].apply(wkt.loads)
            gdf = gdf.set_geometry("geometry")
            return gdf
