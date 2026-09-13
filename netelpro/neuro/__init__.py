"""Netelpro Neuro: Neuro-symbolic activation and deterministic gating subsystem.

Bridges neural representations with compiled formal logic verification.
"""

from __future__ import annotations

from netelpro.neuro.neuron import NetelproLayer, NetelproNeuron
from netelpro.neuro.ste import NetelproActivationSTE
from netelpro.neuro.logits_gate import NetelproLogitsGate
from netelpro.neuro.logits_processor import NetelproLogitsProcessor
from netelpro.neuro.native_kernel import NetelproVectorKernel
from netelpro.neuro.neuro_gate import NetelproNeuroGate
from netelpro.neuro.certificate import AuditCertificate, LayerAuditRecord
from netelpro.neuro.network import NetelproDeepNetwork

__all__ = [
    "NetelproNeuron",
    "NetelproLayer",
    "NetelproActivationSTE",
    "NetelproLogitsGate",
    "NetelproLogitsProcessor",
    "NetelproNeuroGate",
    "NetelproVectorKernel",
    "AuditCertificate",
    "LayerAuditRecord",
    "NetelproDeepNetwork",
]
