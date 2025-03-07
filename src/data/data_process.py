import polars as pl
import requests
import os
import logging
import geopandas as gpd
from shapely import wkt
from ..jp_qcew.src.data.data_process import cleanData
from ..models import init_zipstable


class foodDeseart(cleanData):
    def __init__(
        self,
        saving_dir: str = "data/",
        database_file: str = "data.ddb",
        log_file: str = "data_process.log",
    ):
        super().__init__(saving_dir, database_file, log_file)

    def buiss_data(self, year: int) -> gpd.GeoDataFrame:
        df = self.conn.sql(f"SELECT * FROM 'qcewtable' WHERE year = {year};").pl()
        codes = [
            "4451",
            "44511",
            "445110",
            "44513",
            "445131",
            "445132",
            "4452",
            "44523",
            "445230",
            "44524",
            "445240",
            "445240",
            "44525",
            "445250",
            "44529",
            "445291",
            "445292",
            "445298",
            "4551",
            "45511",
            "455110",
            "4552",
            "45521",
            "455211",
            "455219",
        ]
        df = df.select(
            pl.col(
                "year",
                "qtr",
                "phys_addr_5_zip",
                "naics_code",
            )
        )

        df = df.with_columns(
            zipcode=pl.col("phys_addr_5_zip").cast(pl.String).str.zfill(5),
            naics4=pl.col("naics_code").str.slice(0, 4),
            dummy=pl.lit(1),
        )
        df = df.filter(pl.col("phys_addr_5_zip") != "")
        df = df.filter(pl.col("naics_code") != "")
        df = df.filter(pl.col("naics4").is_in(codes))
        df = df.group_by(["year", "qtr", "naics4", "zipcode"]).agg(
            buisnesses=pl.col("dummy").sum()
        )
        gdf = gpd.read_file(
            f"{self.saving_dir}external/zips_shape.zip", engine="pyogrio"
        )
        gdf = gdf.to_crs("EPSG:4326")
        gdf = gdf.rename(columns={"ZCTA5CE20": "zipcode"})
        gdf = gdf[gdf["zipcode"].astype(int) < 999].reset_index()
        gdf = gdf[["zipcode", "geometry"]]

        gdf = gdf.join(df.to_pandas(), on="zipcode", how="inner")

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
