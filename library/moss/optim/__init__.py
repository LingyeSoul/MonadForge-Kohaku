"""Optimisers extracted from KohakuMossTrainer.

``AdamW16`` keeps the parameter and both moments in the parameter's own 16-bit dtype
with fp32 math in registers; ``MuonHybrid`` runs Muon (Nesterov momentum plus a
Newton-Schulz orthogonalisation) on 2-D groups and AdamW on the rest. Vendored from
https://github.com/KohakuBlueleaf/KohakuMossTrainer under the KohakuMoss License 1.0
(``library/moss/vendor/LICENSE-KohakuMoss``); imports and docs pointers changed only.
"""

from library.moss.optim.adamw16 import AdamW16
from library.moss.optim.muon import MuonHybrid, newton_schulz

__all__ = ["AdamW16", "MuonHybrid", "newton_schulz"]
