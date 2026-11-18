# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import os
import sys

try:
    from .version import __version__  # noqa
except ImportError:
    version_txt = os.path.join(os.path.dirname(__file__), "version.txt")
    with open(version_txt) as f:
        __version__ = f.read().strip()

__all__ = ["pdb"]

# backwards compatibility to support `from fairseq.X import Y`
from fairseq.distributed import utils as distributed_utils
from fairseq.logging import meters, metrics, progress_bar  # noqa

sys.modules["fairseq.distributed_utils"] = distributed_utils
sys.modules["fairseq.meters"] = meters
sys.modules["fairseq.metrics"] = metrics
sys.modules["fairseq.progress_bar"] = progress_bar

# initialize hydra
from fairseq.dataclass.initialize import hydra_init

try:
    hydra_init()
except Exception:
    pass

import importlib
for _mod in [
    "fairseq.criterions",
    "fairseq.distributed",
    "fairseq.models",
    "fairseq.modules",
    "fairseq.optim",
    "fairseq.optim.lr_scheduler",
    "fairseq.pdb",
    "fairseq.scoring",
    "fairseq.tasks",
    "fairseq.token_generation_constraints",
]:
    try:
        importlib.import_module(_mod)
    except Exception:
        pass

try:
    import fairseq.benchmark  # noqa
    import fairseq.model_parallel  # noqa
except Exception:
    pass
