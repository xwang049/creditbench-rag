"""Query fundamentals parquet files using DuckDB.

Automatically uses Azure Blob Storage if AZURE_STORAGE_ACCOUNT and
AZURE_STORAGE_SAS_TOKEN are set in .env, otherwise falls back to local
OneDrive mount path.
"""

import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.config import config

logger = logging.getLogger(__name__)

# Short name → parquet filename
FUNDAMENTALS_FILES = {
    "af":      "fundamentals_namr_af_history.parquet",
    "bs":      "fundamentals_namr_bs_history.parquet",
    "cf":      "fundamentals_namr_cf_history.parquet",
    "cr2":     "fundamentals_namr_cr2_history.parquet",
    "ind1":    "fundamentals_namr_ind1_history.parquet",
    "ind2":    "fundamentals_namr_ind2_history.parquet",
    "ind3":    "fundamentals_namr_ind3_history.parquet",
    "ind4":    "fundamentals_namr_ind4_history.parquet",
    "is_":     "fundamentals_namr_is_history.parquet",
    "sard_bs": "fundamentals_namr_sard_bs_history.parquet",
    "sard_cf": "fundamentals_namr_sard_cf_history.parquet",
    "sard_is": "fundamentals_namr_sard_is_history.parquet",
}


def _build_connection() -> tuple[duckdb.DuckDBPyConnection, dict[str, str]]:
    """Create a DuckDB connection and return (conn, {name: parquet_path}).

    Paths are Azure URLs when Azure config is present, otherwise local paths.
    """
    con = duckdb.connect()

    if config.use_azure:
        logger.debug("Using Azure Blob Storage for fundamentals")
        con.execute("INSTALL azure; LOAD azure;")
        # SAS token from .env may start with '?', strip it for connection string
        sas = config.AZURE_STORAGE_SAS_TOKEN.lstrip("?")
        conn_str = f"AccountName={config.AZURE_STORAGE_ACCOUNT};SharedAccessSignature={sas}"
        con.execute(f"CREATE SECRET azure_secret (TYPE azure, CONNECTION_STRING '{conn_str}');")

        prefix = config.AZURE_FUNDAMENTALS_PATH.strip("/")
        base = f"azure://{config.AZURE_BLOB_CONTAINER}" + (f"/{prefix}" if prefix else "")
        paths = {name: f"{base}/{filename}" for name, filename in FUNDAMENTALS_FILES.items()}
    else:
        logger.debug("Using local parquet files for fundamentals")
        local_dir = Path(config.DATA_DIR) / "foundamentals"
        paths = {
            name: str(local_dir / filename)
            for name, filename in FUNDAMENTALS_FILES.items()
            if (local_dir / filename).exists()
        }

    return con, paths


def _register_views(con: duckdb.DuckDBPyConnection, paths: dict[str, str]) -> None:
    for name, path in paths.items():
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path}')")


def query(sql: str) -> pd.DataFrame:
    """Run SQL against fundamentals parquet files.

    Table names: af, bs, cf, cr2, ind1, ind2, ind3, ind4, is_, sard_bs, sard_cf, sard_is

    Example:
        df = query(\"\"\"
            SELECT ID_BB_COMPANY, FISCAL_YEAR_PERIOD, IS_OPER_INC
            FROM is_
            WHERE FISCAL_YEAR_PERIOD >= '2020'
            LIMIT 100
        \"\"\")
    """
    con, paths = _build_connection()
    _register_views(con, paths)
    result = con.execute(sql).df()
    con.close()
    return result


def list_columns(table: str) -> list[str]:
    """List all columns in a fundamentals table."""
    if table not in FUNDAMENTALS_FILES:
        raise ValueError(f"Unknown table '{table}'. Available: {list(FUNDAMENTALS_FILES.keys())}")

    con, paths = _build_connection()
    if table not in paths:
        raise FileNotFoundError(f"Parquet file for '{table}' not found")

    con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{paths[table]}')")
    cols = con.execute(f"DESCRIBE {table}").df()["column_name"].tolist()
    con.close()
    return cols


def data_source() -> str:
    """Return a string describing where fundamentals data is being read from."""
    if config.use_azure:
        return (
            f"Azure Blob  "
            f"account={config.AZURE_STORAGE_ACCOUNT}  "
            f"container={config.AZURE_BLOB_CONTAINER}/{config.AZURE_FUNDAMENTALS_PATH}"
        )
    return f"Local  path={Path(config.DATA_DIR) / 'foundamentals'}"
