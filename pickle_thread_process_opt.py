from cProfile import run
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from os.path import join
from pathlib import Path
from pickle import dump, dumps, load
from pickletools import optimize
from random import seed, uniform
from shutil import copy, rmtree

seed(42)
test_folder = Path("test")
rmtree(test_folder, ignore_errors=True)
test_folder.mkdir(exist_ok=True)
obj = [[[uniform(-100, 100) for _ in range(350)] for _ in range(350)] for _ in range(10)]
test = {}
for dl in ["_dump", "_load"]:
    for tp in ["_thread", "_process"]:
        for o in ["", "_opt"]:
            folder = f"test{dl}{tp}{o}"
            folder_path = Path(join(test_folder, folder))
            folder_path.mkdir(exist_ok = True)
            test[folder] = [join(folder_path, str(i)) for i in range(100)]

def dump_func(path):
    with open(path, "wb") as f:
        dump(obj, f)

def dump_optimize_func(path):
    with open(path, "wb") as f:
        f.write(optimize(dumps(obj)))

def load_func(path):
    with open(path, "rb") as f:
        load(f)

def test_dump_thread(pool: ThreadPool):
    pool.map(dump_func, test["test_dump_thread"])

def test_dump_process(pool: Pool):
    pool.map(dump_func, test["test_dump_process"])

def test_load_thread(pool: ThreadPool):
    pool.map(load_func, test["test_load_thread"])

def test_load_process(pool: Pool):
    pool.map(load_func, test["test_load_process"])

def test_dump_thread_opt(pool: ThreadPool):
    pool.map(dump_func, test["test_dump_thread_opt"])

def test_dump_process_opt(pool: Pool):
    pool.map(dump_func, test["test_dump_process_opt"])

def test_load_thread_opt(pool: ThreadPool):
    pool.map(load_func, test["test_load_thread_opt"])

def test_load_process_opt(pool: Pool):
    pool.map(load_func, test["test_load_process_opt"])

def main():
    with ThreadPool(16) as pool:
        test_dump_thread(pool)
        for f1, f2 in zip(test["test_dump_thread"], test["test_load_thread"]):
            copy(f1, f2)
        test_load_thread(pool)

    with Pool(16) as pool:
        test_dump_process(pool)
        for f1, f2 in zip(test["test_dump_process"], test["test_load_process"]):
            copy(f1, f2)
        test_load_process(pool)

    with ThreadPool(16) as pool:
        test_dump_thread_opt(pool)
        for f1, f2 in zip(test["test_dump_thread_opt"], test["test_load_thread_opt"]):
            copy(f1, f2)
        test_load_thread_opt(pool)

    with Pool(16) as pool:
        test_dump_process_opt(pool)
        for f1, f2 in zip(test["test_dump_process_opt"], test["test_load_process_opt"]):
            copy(f1, f2)
        test_load_process_opt(pool)

    rmtree(test_folder, ignore_errors=True)

run("main()", "test.pstats")