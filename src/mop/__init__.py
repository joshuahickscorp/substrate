__version__ = "0.1.0"

_runtime_process_label = _os.environ.get("MOP_PROCESS_LABEL")
if _runtime_process_label:
    from mop.process_labels import set_process_label as _set_process_label

    _set_process_label(_runtime_process_label)

del _os, _runtime_process_label
if "_set_process_label" in globals():
    del _set_process_label
