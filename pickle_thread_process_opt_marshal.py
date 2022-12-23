from cProfile import run
from marshal import dump as marshal_dump, load as marshal_load
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from os.path import join
from pathlib import Path
from pickle import dump as pickle_dump, dumps as pickle_dumps, load as pickle_load
from pickletools import optimize
from random import seed, uniform
from shutil import copy, rmtree

seed(42)
num_files = 256
num_workers = 16
objs = [[[[uniform(-100, 100) for _ in range(350)] for _ in range(350)] for _ in range(10)] for _ in range(num_files)]

test_folder = Path("test")
rmtree(test_folder, ignore_errors=True)
test_folder.mkdir(exist_ok=True)
test = {}
for dl in ["_dump", "_load"]:
    for tp in ["", "_thread", "_process"]:
        for o in ["", "_opt", "_marshal"]:
            folder = f"test{o}{dl}{tp}"
            folder_path = Path(join(test_folder, folder))
            folder_path.mkdir(exist_ok = True)
            test[folder] = [join(folder_path, str(i)) for i in range(num_files)]

def dump_func(obj, path):
    with open(path, "wb") as f:
        pickle_dump(obj, f)

def load_func(path):
    with open(path, "rb") as f:
        pickle_load(f)

def opt_dump_func(obj, path):
    with open(path, "wb") as f:
        f.write(optimize(pickle_dumps(obj)))

opt_load_func = load_func

def marshal_dump_func(obj, path):
    with open(path, "wb") as f:
        marshal_dump(obj, f)

def marshal_load_func(path):
    with open(path, "rb") as f:
        marshal_load(f)

def test_dump(obj):
    for obj, path in zip(objs, test["test_dump"]):
        dump_func(obj, path)

def test_dump_thread(pool: ThreadPool):
    pool.starmap(dump_func, zip(objs, test["test_dump_thread"]))

def test_dump_process(pool: Pool):
    pool.starmap(dump_func, zip(objs, test["test_dump_process"]))

def test_load():
    for path in test["test_load"]:
        load_func(path)

def test_load_thread(pool: ThreadPool):
    pool.map(load_func, test["test_load_thread"])

def test_load_process(pool: Pool):
    pool.map(load_func, test["test_load_process"])

def test_opt_dump():
    for obj, path in zip(objs, test["test_opt_dump"]):
        opt_dump_func(obj, path)

def test_opt_dump_thread(pool: ThreadPool):
    pool.starmap(opt_dump_func, zip(objs, test["test_opt_dump_thread"]))

def test_opt_dump_process(pool: Pool):
    pool.starmap(opt_dump_func, zip(objs, test["test_opt_dump_process"]))

def test_opt_load():
    for path in test["test_opt_load"]:
        opt_load_func(path)

def test_opt_load_thread(pool: ThreadPool):
    pool.map(opt_load_func, test["test_opt_load_thread"])

def test_opt_load_process(pool: Pool):
    pool.map(opt_load_func, test["test_opt_load_process"])

def test_marshal_dump():
    for obj, path in zip(objs, test["test_marshal_dump"]):
        marshal_dump_func(obj, path)

def test_marshal_dump_thread(pool: ThreadPool):
    pool.starmap(marshal_dump_func, zip(objs, test["test_marshal_dump_thread"]))

def test_marshal_dump_process(pool: Pool):
    pool.starmap(marshal_dump_func, zip(objs, test["test_marshal_dump_process"]))

def test_marshal_load():
    for path in test["test_marshal_load"]:
        marshal_load_func(path)

def test_marshal_load_thread(pool: ThreadPool):
    pool.map(marshal_load_func, test["test_marshal_load_thread"])

def test_marshal_load_process(pool: Pool):
    pool.map(marshal_load_func, test["test_marshal_load_process"])

def main():
    # test_dump()
    # for f1, f2 in zip(test["test_dump"], test["test_load"]):
    #     copy(f1, f2)
    # test_load()

    # with ThreadPool(num_workers) as pool:
    #     test_dump_thread(pool)
    #     for f1, f2 in zip(test["test_dump_thread"], test["test_load_thread"]):
    #         copy(f1, f2)
    #     test_load_thread(pool)

    with Pool(num_workers) as pool:
        test_dump_process(pool)
        for f1, f2 in zip(test["test_dump_process"], test["test_load_process"]):
            copy(f1, f2)
        test_load_process(pool)

    # test_opt_dump()
    # for f1, f2 in zip(test["test_opt_dump"], test["test_opt_load"]):
    #     copy(f1, f2)
    # test_opt_load()

    # with ThreadPool(num_workers) as pool:
    #     test_opt_dump_thread(pool)
    #     for f1, f2 in zip(test["test_opt_dump_thread"], test["test_opt_load_thread"]):
    #         copy(f1, f2)
    #     test_opt_load_thread(pool)

    # with Pool(num_workers) as pool:
    #     test_opt_dump_process(pool)
    #     for f1, f2 in zip(test["test_opt_dump_process"], test["test_opt_load_process"]):
    #         copy(f1, f2)
    #     test_opt_load_process(pool)

    # test_marshal_dump()
    # for f1, f2 in zip(test["test_marshal_dump"], test["test_marshal_load"]):
    #     copy(f1, f2)
    # test_marshal_load()

    # with ThreadPool(num_workers) as pool:
    #     test_marshal_dump_thread(pool)
    #     for f1, f2 in zip(test["test_marshal_dump_thread"], test["test_marshal_load_thread"]):
    #         copy(f1, f2)
    #     test_marshal_load_thread(pool)

    # with Pool(num_workers) as pool:
    #     test_marshal_dump_process(pool)
    #     for f1, f2 in zip(test["test_marshal_dump_process"], test["test_marshal_load_process"]):
    #         copy(f1, f2)
    #     test_marshal_load_process(pool)

    rmtree(test_folder, ignore_errors=True)

run("main()", "test.pstats")