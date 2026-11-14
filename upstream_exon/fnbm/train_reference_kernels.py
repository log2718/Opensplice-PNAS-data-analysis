#!/usr/bin/env python
"""Run Arush's train.py with the Triton fast paths switched off.

Triton imports cleanly in our venv but dies at driver initialisation: the first
kernel launch JIT-compiles cuda_utils.so with gcc, which needs the CUDA headers
and libcuda stub that his jobs inherit from the cuda12.1.1-cudnn8.9.0-devel
singularity image (see jobs/alphagenome_splicing/train.sbatch in his repo).
Bare metal on Torch has neither, so jobs 22861921/22861922 both died ~60 s in
with CalledProcessError out of triton/runtime/build.py.

Every kernel has a plain-PyTorch reference, chosen per module by a
``use_kernels`` class attribute that ``forward`` reads at call time. Setting it
False is his own documented escape hatch (models/blocks.py:249) and exactly
what scripts/ablate_clusters.py does. The reference path is slower and keeps the
(B*F*L, K) hidden-unit tensor resident -- at batch 128, 64 filters, L=113 and
K=32 that is about 118 MB, which an L40S absorbs without noticing.

HIS REPOSITORY IS NOT MODIFIED. This sets two class attributes in the imported
modules and then hands control to his unmodified train.py.

Order 1 only reaches BasisFilterTransform. BasisInteractionBlock is switched
off too, so an order-2 follow-up does not rediscover this failure.

    python upstream_exon/fnbm/train_reference_kernels.py <FNBM_REPO> <CONFIG>
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    fnbm = Path(sys.argv[1]).resolve()
    config = str(Path(sys.argv[2]).resolve())

    if not (fnbm / "train.py").exists():
        sys.exit(f"FATAL: {fnbm}/train.py not found")

    # Python puts THIS script's directory on sys.path, not his checkout, and
    # train.py resolves its own imports relative to the repo root.
    sys.path.insert(0, str(fnbm))
    os.chdir(fnbm)

    from models.blocks import BasisFilterTransform
    from models.fnbm.interaction_block import BasisInteractionBlock

    BasisFilterTransform.use_kernels = False
    BasisInteractionBlock.use_kernels = False
    print("triton fast paths DISABLED -> forward_reference "
          f"(BasisFilterTransform.use_kernels="
          f"{BasisFilterTransform.use_kernels})", flush=True)

    sys.argv = ["train.py", config]
    runpy.run_path(str(fnbm / "train.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
