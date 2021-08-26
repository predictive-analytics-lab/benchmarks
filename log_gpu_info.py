"""Query nvidia-smi and save the result to an SQLite database."""
from __future__ import annotations
import csv
import datetime
from io import StringIO
import sqlite3
import subprocess
import time
from typing import Final

QUERIES: Final = [
    "timestamp",
    "gpu_name",
    "gpu_bus_id",
    "pstate",
    "clocks_throttle_reasons.hw_thermal_slowdown",
    "memory.used",
    "utilization.gpu",
    "utilization.memory",
    "temperature.gpu",
    "power.draw",
    "fan.speed",
]
DB_FILE_NAME: Final = "gpu_info.db"
INTERVAL: Final = 5.0  # in seconds


def main() -> None:
    db_cur, db_con = connect_db()
    try:
        while True:
            rows = get_new_rows()  # from nvidia-smi
            insert_values(db_cur, rows)
            db_con.commit()
            time.sleep(INTERVAL)
    finally:
        db_con.close()


def get_new_rows() -> list[dict]:
    """Call nvidia-smi to get the info."""
    output = subprocess.run(
        [
            "/usr/bin/nvidia-smi",
            "--query-gpu=" + ",".join(QUERIES),
            "--format=csv",
        ],
        check=True,
        capture_output=True,
    ).stdout
    f = StringIO(output.decode("utf-8"))
    reader = csv.DictReader(f, delimiter=",", skipinitialspace=True)
    rows = list(reader)
    for row in rows:
        parse(row)
    return rows


def parse(row: dict) -> None:
    """Remove units and convert to the right data type."""
    row["timestamp"] = datetime.datetime.fromisoformat(
        row["timestamp"].replace("/", "-")
    ).timestamp()
    # "gpu_name"
    # "gpu_bus_id"
    # "pstate"
    # "clocks_throttle_reasons.hw_thermal_slowdown"
    row["memory.used [MiB]"] = float(row["memory.used [MiB]"].rstrip(" MiB"))
    row["utilization.gpu [%]"] = float(row["utilization.gpu [%]"].rstrip(" %"))
    row["utilization.memory [%]"] = float(row["utilization.memory [%]"].rstrip(" %"))
    row["temperature.gpu"] = float(row["temperature.gpu"])
    row["power.draw [W]"] = float(row["power.draw [W]"].rstrip(" W"))
    row["fan.speed [%]"] = float(row["fan.speed [%]"].rstrip(" %"))


def connect_db() -> tuple[sqlite3.Cursor, sqlite3.Connection]:
    con = sqlite3.connect(DB_FILE_NAME)
    cur = con.cursor()
    cur.execute(
        """CREATE TABLE IF NOT EXISTS gpu_data
                   (timestamp REAL,
                    name TEXT,
                    pci_bus_id TEXT,
                    pstate TEXT,
                    clocks_throttle_reasons_hw_thermal_slowdown TEXT,
                    "memory_used [MiB]" REAL,
                    "utilization_gpu [%]" REAL,
                    "utilization_memory [%]" REAL,
                    "temperature_gpu" REAL,
                    "power_draw [W]" REAL,
                    "fan_speed [%]" REAL)"""
    )
    return cur, con


def insert_values(cur: sqlite3.Cursor, rows):
    cur.executemany(
        "INSERT INTO gpu_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [tuple(row.values()) for row in rows],
    )


if __name__ == "__main__":
    main()
