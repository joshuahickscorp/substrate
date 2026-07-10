from concurrent.futures import ThreadPoolExecutor

from mop.logging_utils import new_run_dir


def test_new_run_dir_is_unique_under_concurrency(tmp_path):
    def allocate(_: int):
        return new_run_dir("parallel", root=tmp_path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(allocate, range(24)))

    assert len(paths) == len(set(paths)) == 24
    assert all(path.is_dir() for path in paths)
