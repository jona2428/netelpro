"""Tests for NetelproDeepNetwork and AuditCertificate."""

from __future__ import annotations

from pathlib import Path
import sys
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from netelpro.neuro import NetelproDeepNetwork, AuditCertificate


def test_deep_network_forward_and_certificate():
    # 4 inputs -> 8 hidden -> 4 hidden -> 2 outputs
    net = NetelproDeepNetwork(
        layer_dims=[4, 8, 4, 2],
        scale_factor=1000.0,
        z_min=-5000,
        z_max=5000,
        activation_fn="relu",
    )

    x = torch.randn(2, 4)
    logits, cert = net(x, control_flags=1)

    assert logits.shape == (2, 2)
    assert isinstance(cert, AuditCertificate)
    assert cert.total_neurons_audited == (8 + 4)
    assert cert.is_fully_compliant is True
    assert cert.total_latency_us > 0.0

    summary = cert.summary()
    assert "Netelpro Formal Audit Certificate" in summary
    assert "Layer 1" in summary
    assert "Layer 2" in summary


def test_deep_network_backprop_training():
    net = NetelproDeepNetwork(
        layer_dims=[3, 6, 2],
        scale_factor=1000.0,
        z_min=-10000,
        z_max=10000,
        activation_fn="relu",
    )
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    initial_params = [p.clone().detach() for p in net.parameters()]

    x = torch.randn(4, 3)
    target = torch.tensor([0, 1, 0, 1])
    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()
    logits, cert = net(x, control_flags=1)
    loss = criterion(logits, target)
    loss.backward()

    # Check gradients in both hidden layer and output layer
    for p in net.parameters():
        assert p.grad is not None

    optimizer.step()

    # Verify that weights were updated across layers
    has_moved = False
    for p_init, p_curr in zip(initial_params, net.parameters()):
        if not torch.allclose(p_init, p_curr):
            has_moved = True
            break
    assert has_moved, "Parameters should have updated through backprop"


def test_deep_network_emergency_inhibition():
    net = NetelproDeepNetwork(
        layer_dims=[4, 8, 2],
        scale_factor=1000.0,
        z_min=-5000,
        z_max=5000,
        activation_fn="relu",
    )

    x = torch.randn(1, 4)
    # Freeze with control_flags = 0
    logits, cert = net(x, control_flags=0)

    # First layer (8 neurons) must all be suppressed
    assert cert.records[0].suppressed_neurons == 8
    assert cert.records[0].active_neurons == 0
