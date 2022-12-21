from cProfile import run
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from os.path import join
from pathlib import Path
from pickle import dump, load
from shutil import copy, rmtree

test_folder = Path("test")
rmtree(test_folder, ignore_errors=True)
test_folder.mkdir(exist_ok=True)
obj = [list(map(float, range(350))) * 350] * 100
test = {}
for dl in ["dump", "load"]:
    for tp in ["thread", "process"]:
        folder = f"test_{dl}_{tp}"
        folder_path = Path(join(test_folder, folder))
        folder_path.mkdir(exist_ok = True)
        test[folder] = [join(folder_path, str(i)) for i in range(100)]

def dump_func(path):
    with open(path, "wb") as f:
        dump(obj, f)

def load_func(path):
    with open(path, "rb") as f:
        load(f)

def test_thread_dump(pool: ThreadPool):
    pool.map(dump_func, test["test_dump_thread"])

def test_process_dump(pool: Pool):
    pool.map(dump_func, test["test_dump_process"])

def test_thread_load(pool: ThreadPool):
    pool.map(load_func, test["test_load_thread"])

def test_process_load(pool: Pool):
    pool.map(load_func, test["test_load_process"])

def main():
    with ThreadPool(16) as pool:
        test_thread_dump(pool)
        for f1, f2 in zip(test["test_dump_thread"], test["test_load_thread"]):
            copy(f1, f2)
        test_thread_load(pool)

    with Pool(16) as pool:
        test_process_dump(pool)
        for f1, f2 in zip(test["test_dump_process"], test["test_load_process"]):
            copy(f1, f2)
        test_process_load(pool)

    rmtree(test_folder, ignore_errors=True)

run("main()", "test.pstats")