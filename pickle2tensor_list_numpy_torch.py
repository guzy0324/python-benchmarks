from cProfile import run
from multiprocessing import Pool
from os.path import join
from pathlib import Path
from pickle import dump as pickle_dump, load as pickle_load
from random import seed, uniform
from shutil import copy, rmtree

from numpy import load as numpy_load, save as numpy_save
from torch import float32, load as torch_load, save as torch_save, tensor

seed(42)
num_files = 32
num_workers = 16
objs = tensor([[[[uniform(-100, 100) for _ in range(350)] for _ in range(350)] for _ in range(20)] for _ in range(num_files)], dtype=float32)

test_folder = Path("test")
rmtree(test_folder, ignore_errors=True)
test_folder.mkdir(exist_ok=True)
test = {}
for dl in ["_dump", "_load"]:
    for tp in ["", "_process"]:
        for t in ["_list", "_numpy", "_torch"]:
            folder = f"test{dl}{tp}{t}"
            folder_path = Path(join(test_folder, folder))
            folder_path.mkdir(exist_ok = True)
            test[folder] = [join(folder_path, str(i)) for i in range(num_files)]

def dump_list_func(obj, path):
    with open(path, "wb") as f:
        pickle_dump(obj.tolist(), f)

def load_list_func(path):
    with open(path, "rb") as f:
        tensor(pickle_load(f), dtype=float32)

def dump_numpy_func(obj, path):
    with open(path, "wb") as f:
        numpy_save(f, obj)

def load_numpy_func(path):
    with open(path, "rb") as f:
        tensor(numpy_load(f), dtype=float32)

def dump_torch_func(obj, path):
    with open(path, "wb") as f:
        torch_save(obj, f)

def load_torch_func(path):
    with open(path, "rb") as f:
        torch_load(f)

def test_dump_list():
    dump_list_func(objs[0], test["test_dump_list"][0])

def test_load_list():
    load_list_func(test["test_load_list"][0])

def test_dump_numpy():
    dump_numpy_func(objs[0], test["test_dump_numpy"][0])

def test_load_numpy():
    load_numpy_func(test["test_load_numpy"][0])

def test_dump_torch():
    dump_torch_func(objs[0], test["test_dump_torch"][0])

def test_load_torch():
    load_torch_func(test["test_load_torch"][0])

def test_dump_process_list(pool: Pool):
    pool.starmap(dump_list_func, zip(objs, test["test_dump_process_list"]))

def test_load_process_list(pool: Pool):
    pool.map(load_list_func, test["test_load_process_list"])

def test_dump_process_numpy(pool: Pool):
    pool.starmap(dump_numpy_func, zip(objs, test["test_dump_process_numpy"]))

def test_load_process_numpy(pool: Pool):
    pool.map(load_numpy_func, test["test_load_process_numpy"])

def test_dump_process_torch(pool: Pool):
    pool.starmap(dump_torch_func, zip(objs, test["test_dump_process_torch"]))

def test_load_process_torch(pool: Pool):
    pool.map(load_torch_func, test["test_load_process_torch"])

def main():
    test_dump_list()
    copy(test["test_dump_list"][0], test["test_load_list"][0])
    test_load_list()

    test_dump_numpy()
    copy(test["test_dump_numpy"][0], test["test_load_numpy"][0])
    test_load_numpy()

    test_dump_torch()
    copy(test["test_dump_torch"][0], test["test_load_torch"][0])
    test_load_torch()

    with Pool(num_workers) as pool:
        test_dump_process_list(pool)
        for f1, f2 in zip(test["test_dump_process_list"], test["test_load_process_list"]):
            copy(f1, f2)
        test_load_process_list(pool)

    with Pool(num_workers) as pool:
        test_dump_process_numpy(pool)
        for f1, f2 in zip(test["test_dump_process_numpy"], test["test_load_process_numpy"]):
            copy(f1, f2)
        test_load_process_numpy(pool)

    with Pool(num_workers) as pool:
        test_dump_process_torch(pool)
        for f1, f2 in zip(test["test_dump_process_torch"], test["test_load_process_torch"]):
            copy(f1, f2)
        test_load_process_torch(pool)

    rmtree(test_folder, ignore_errors=True)

run("main()", "test.pstats")