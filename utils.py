# utils.py
import time, os
from contextlib import contextmanager
from pyspark.sql import SparkSession

@contextmanager
def timer(msg):
    t0 = time.time()
    print(f"[TIMER] {msg} ...")
    yield
    dt = time.time() - t0
    print(f"[TIMER] {msg} done in {dt:.2f}s")

def make_spark(app, master=None, extra_conf=None):
    b = SparkSession.builder.appName(app)
    if master:
        b = b.master(master)
    if extra_conf:
        for k,v in extra_conf.items():
            b = b.config(k, v)
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

def ensure_dir(path):
    if path.startswith("hdfs://"):  # leave to HDFS
        return path
    os.makedirs(path, exist_ok=True); return path
