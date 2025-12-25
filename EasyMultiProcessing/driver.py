import multiprocessing as mp
import os
from typing import Callable, Dict, Any
from math import ceil
from functools import partial
from .file import (
    load_json,
    save_json,
    load_pickle,
    save_pickle,
    gather_subprocess_cache,
    subprocess_filepath,
)

def sub_iterable_params(iterable_params: Dict[str, Any], rank: int, n_iter: int):
    return {k: v[rank*n_iter: (rank+1)*n_iter] for k, v in iterable_params.items()}

def sub_modulo_params(modulo_params: Dict[str, Any], rank: int):
    return {k: v[rank % len(v)] for k, v in modulo_params.items()}

def sub_singleton_params(singleton_params: Dict[str, Any], rank: int):
    return {k: v[1 if rank else 0] for k, v in singleton_params.items()}

def function_wrapper2(kwargs):
    function_wrapper(**kwargs)

def function_wrapper(
        rank: int,
        func: Callable,
        save_filepath: str,
        not_iterable_params: Dict[str, Any],
        singleton_params: Dict[str, Any],
        modulo_params: Dict[str, Any],
        iterable_params: Dict[str, Any],
):
    rank_result = func(**not_iterable_params, **singleton_params, **modulo_params, **iterable_params)
    rank_filepath = subprocess_filepath(rank, save_filepath)
    if rank_filepath.endswith(".json"):
        try:
            save_json(rank_result, rank_filepath)
        except TypeError:
            save_pickle(rank_result, rank_filepath[:-5] + ".pickle")
    else:
        assert rank_filepath.endswith(".pickle")
        save_pickle(rank_result, rank_filepath)

def multiprocessing_wrapper(
        func: Callable,
        nproc: int,
        save_filepath: str,
        not_iterable_params: Dict[str, Any] = None,
        singleton_params: Dict[str, Any] = None,
        modulo_params: Dict[str, Any] = None,
        iterable_params: Dict[str, Any] = None,
        overwrite = False,
):
    """
    Wrapper for mapping single processing function to multiprocessing
    :param func:
    :param nproc:
    :param save_filepath:
    :param not_iterable_params: such as {'hoi': Ture, 'num_pairs': 100}
    :param singleton_params: such as {'show_progress_bar': [True, False]}
    :param modulo_params: such as {'gpu_id': [1, 2, 3, 4, 5, 0, 6, 7]}
    :param iterable_params:
    :param overwrite:
    :return:
    """
    mp.set_start_method("spawn", force=True)
    save_name, save_ext = os.path.splitext(save_filepath)
    if os.path.exists(save_name + ".json"):
        if overwrite:
            os.remove(save_name + ".json")
        else:
            return load_json(save_name + ".json")
    elif os.path.exists(save_name + ".pickle"):
        if overwrite:
            os.remove(save_name + ".pickle")
        else:
            return load_pickle(save_name + ".pickle")
    else:
        os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
    n_iter, n_iter_per_proc = None, 0
    if iterable_params:
        for v in iterable_params.values():
            if n_iter is None:
                n_iter = len(v)
            else:
                assert n_iter == len(v)
        n_iter_per_proc = ceil(n_iter / nproc)
    if not_iterable_params is None:
        not_iterable_params = {}
    pool = []
    for rank in range(nproc):
        if singleton_params is not None:
            rank_singleton_params = sub_singleton_params(singleton_params, rank)
        else:
            rank_singleton_params = {}
        if n_iter is not None:
            rank_iterable_params = sub_iterable_params(iterable_params, rank, n_iter_per_proc)
        else:
            rank_iterable_params = {}
        if modulo_params is not None:
            rank_modulo_params = sub_modulo_params(modulo_params, rank)
        else:
            rank_modulo_params = {}
        pool.append(dict(
            rank=rank,
            func=func,
            save_filepath=save_filepath,
            not_iterable_params=not_iterable_params,
            singleton_params=rank_singleton_params,
            modulo_params=rank_modulo_params,
            iterable_params=rank_iterable_params,
        ))
    if nproc > 1:
        with mp.Pool(nproc) as p:
            p.map(function_wrapper2, pool)
    else:
        function_wrapper(**pool[0])
    gathered_results = gather_subprocess_cache(nproc, save_filepath)
    return gathered_results
