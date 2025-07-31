import json
import os
import pickle

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def subprocess_filepath(rank, filepath):
    n, e = os.path.splitext(filepath)
    return f"{n}.{rank}{e}"

def gather_subprocess_cache(nproc, filepath):
    gathered = []
    pickle_mark = filepath.endswith(".pickle")
    caches = []
    for i in range(nproc):
        sub_fp = subprocess_filepath(i, filepath)
        if pickle_mark or not os.path.exists(sub_fp):
            sub_fp_pkl = os.path.splitext(sub_fp)[0] + ".pickle"
            assert os.path.exists(sub_fp_pkl)
            pickle_mark = True
            gathered.extend(load_pickle(sub_fp_pkl))
            caches.append(sub_fp_pkl)
        else:
            gathered.extend(load_json(sub_fp))
            caches.append(sub_fp)
    if pickle_mark:
        save_pickle(gathered, filepath)
    else:
        save_json(gathered, filepath)
    for fp in caches:
        os.remove(fp)
    return gathered
