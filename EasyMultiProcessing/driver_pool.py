import multiprocessing as mp
from typing import Callable, Dict, Any
from math import ceil
from .driver import (
    sub_modulo_params,
    sub_iterable_params,
    sub_singleton_params,
)
from itertools import chain


def worker_function(args):
    func, rank, not_iterable_params, singleton_params, modulo_params, iterable_params = args
    result = func(**not_iterable_params, **singleton_params, **modulo_params, **iterable_params)
    return rank, result


def multiprocessing_wrapper(
        func: Callable,
        nproc: int,
        not_iterable_params: Dict[str, Any] = None,
        singleton_params: Dict[str, Any] = None,
        modulo_params: Dict[str, Any] = None,
        iterable_params: Dict[str, Any] = None,
):
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

    process_args = []
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
        args = (func, rank, not_iterable_params, rank_singleton_params,
                rank_modulo_params, rank_iterable_params)
        process_args.append(args)

    if nproc == 1:
        return worker_function(process_args[0])

    with mp.Pool(nproc) as pool:
        results = pool.map(worker_function, process_args)

    if not results:
        return None
    results.sort(key=lambda x: x[0])
    first_result = None
    rank = 0
    while rank < nproc:
        result = results[rank][1]
        rank += 1
        if result is not None:
            first_result = result
            break
    if isinstance(first_result, dict):
        gathered_results = first_result
        while rank < nproc:
            result = results[rank][1]
            if result is not None:
                gathered_results.update(result)
        return gathered_results
    elif hasattr(first_result, '__iter__'):
        valid_rank = [rank - 1]
        while rank < nproc:
            result = results[rank][1]
            if result is not None:
                valid_rank.append(rank)
            rank += 1
        if len(valid_rank) == 1:
            return first_result
        def all_elements():
            for vr in valid_rank:
                yield results[vr][1]
        return list(chain.from_iterable(all_elements()))
    else:
        return None
