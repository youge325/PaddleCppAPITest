import importlib.abc
import importlib.util
import os
import sys
import types

import paddle

cur_dir = os.path.dirname(os.path.abspath(__file__))
so_path = os.path.join(cur_dir, "repro_ext_pd_.so")


def __bootstrap__():
    assert os.path.exists(so_path)
    # load custom op shared library with abs path
    custom_ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(
        so_path
    )

    if os.name == "nt" or sys.platform.startswith("darwin"):
        # Cpp Extension only support Linux now
        mod = types.ModuleType(__name__)
    else:
        try:
            spec = importlib.util.spec_from_file_location(__name__, so_path)
            assert spec is not None
            mod = importlib.util.module_from_spec(spec)
            assert isinstance(spec.loader, importlib.abc.Loader)
            spec.loader.exec_module(mod)
        except ImportError:
            mod = types.ModuleType(__name__)

    for custom_op in custom_ops:
        setattr(mod, custom_op, eval(custom_op))


__bootstrap__()
