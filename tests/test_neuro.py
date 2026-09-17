"""Comprehensive test suite for Netelpro Neuro subsystem.

Tests formal deterministic activation, fail-closed suppression,
adversarial stability, PyTorch Straight-Through Estimator (STE) training,
and LLM logit pruning.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.optim as optim

from netelpro.neuro import (
    NetelproNeuron,
    NetelproLayer,
    NetelproLogitsGate,
)


def test_neuron_scalar_forward_allowed():
    """Neuron fires normally when potential is within bounds and control is 1."""
    neuron = NetelproNeuron(
        in_features=2,
        scale_factor=1000.0,
        z_min=0,
        z_max=5000,
        activation_fn="relu",
    )
    # Force deterministic weights and bias
    with torch.no_grad():
        neuron.weight.copy_(torch.tensor([1.0, 2.0]))
        neuron.bias.copy_(torch.tensor([0.5]))

    # x = [1.0, 1.0] -> z = 1.0*1.0 + 2.0*1.0 + 0.5 = 3.5 -> scaled: 3500
    x = torch.tensor([1.0, 1.0])
    out = neuron(x, control_flag=1)

    assert math.isclose(out.item(), 3.5, rel_tol=1e-4)
    audit = neuron.audit()
    assert audit["allow"] is True
    assert audit["reason"] is None
    assert audit["z_scaled"] == 3500


def test_neuron_scalar_forward_denied_out_of_bounds():
    """Neuron cuts off (0.0) when potential exceeds stability bound (fail-closed)."""
    neuron = NetelproNeuron(
        in_features=2,
        scale_factor=1000.0,
        z_min=0,
        z_max=5000,
        activation_fn="relu",
    )
    with torch.no_grad():
        neuron.weight.copy_(torch.tensor([5.0, 5.0]))
        neuron.bias.copy_(torch.tensor([0.0]))

    # x = [1.0, 1.0] -> z = 10.0 -> scaled: 10000 > 5000 (exceeds max)
    x = torch.tensor([1.0, 1.0])
    out = neuron(x, control_flag=1)

    assert out.item() == 0.0
    audit = neuron.audit()
    assert audit["allow"] is False
    assert audit["z_scaled"] == 10000


def test_neuron_scalar_forward_denied_control_zero():
    """Neuron inhibits when control signal is 0, regardless of valid potential."""
    neuron = NetelproNeuron(
        in_features=2,
        scale_factor=1000.0,
        z_min=0,
        z_max=5000,
        activation_fn="relu",
    )
    with torch.no_grad():
        neuron.weight.copy_(torch.tensor([1.0, 1.0]))
        neuron.bias.copy_(torch.tensor([0.0]))

    # x = [1.0, 1.0] -> z = 2.0 (in range), but control_flag = 0
    x = torch.tensor([1.0, 1.0])
    out = neuron(x, control_flag=0)

    assert out.item() == 0.0
    audit = neuron.audit()
    assert audit["allow"] is False
    assert audit["control_flag"] == 0


def test_adversarial_comparison_vs_standard_relu():
    """Demonstrates safety divergence: Standard ReLU hallucinates under attack, Netelpro suppresses."""
    # Standard linear + ReLU
    linear = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1.0]]))

    # Netelpro neuron with max bound 5.0 (5000 scaled)
    neuro = NetelproNeuron(1, scale_factor=1000.0, z_min=0, z_max=5000, activation_fn="relu")
    with torch.no_grad():
        neuro.weight.copy_(torch.tensor([1.0]))
        neuro.bias.copy_(torch.tensor([0.0]))

    adversarial_x = torch.tensor([50.0])

    # Uncontrolled neural network explodes
    uncontrolled_output = torch.relu(linear(adversarial_x)).item()
    assert uncontrolled_output == 50.0

    # Netelpro neuron fails-closed to 0.0
    controlled_output = neuro(adversarial_x, control_flag=1).item()
    assert controlled_output == 0.0
    assert neuro.audit()["allow"] is False


def test_pytorch_ste_backward_training():
    """Straight-Through Estimator allows backpropagation and weight updates."""
    neuron = NetelproNeuron(
        in_features=2,
        scale_factor=1000.0,
        z_min=0,
        z_max=10000,
        activation_fn="relu",
    )
    with torch.no_grad():
        neuron.weight.copy_(torch.tensor([0.5, 0.5]))
        neuron.bias.copy_(torch.tensor([0.0]))

    optimizer = optim.SGD(neuron.parameters(), lr=0.1)

    initial_weights = neuron.weight.clone().detach()

    x = torch.tensor([1.0, 2.0])
    target = torch.tensor(1.0)

    # Training step
    optimizer.zero_grad()
    y = neuron(x, control_flag=1)
    loss = (y - target) ** 2
    loss.backward()

    # Gradient must exist and be non-zero
    assert neuron.weight.grad is not None
    assert torch.any(neuron.weight.grad != 0.0)

    optimizer.step()

    # Weights must have moved toward target
    assert not torch.allclose(neuron.weight, initial_weights)


def test_pytorch_ste_gradient_blocked_when_denied():
    """When the gate suppresses activation, gradient is blocked to 0.0."""
    neuron = NetelproNeuron(
        in_features=2,
        scale_factor=1000.0,
        z_min=0,
        z_max=5000,
        activation_fn="relu",
    )
    # Weights that lead to z exceeding z_max
    with torch.no_grad():
        neuron.weight.copy_(torch.tensor([10.0, 10.0]))
        neuron.bias.copy_(torch.tensor([0.0]))

    x = torch.tensor([1.0, 1.0])  # z = 20.0 -> scaled: 20000 > 5000
    target = torch.tensor(1.0)

    y = neuron(x, control_flag=1)
    assert y.item() == 0.0
    loss = (y - target) ** 2
    loss.backward()

    # Gradient must be strictly zero because gate rejected the state
    assert neuron.weight.grad is not None
    assert torch.all(neuron.weight.grad == 0.0)


def test_netelpro_layer_batch_execution():
    """NetelproLayer processes multi-dimensional batches with selective gating."""
    layer = NetelproLayer(
        in_features=3,
        out_features=2,
        scale_factor=1000.0,
        z_min=-2000,
        z_max=5000,
        activation_fn="relu",
    )

    # Batch of 2 items
    x = torch.tensor([[1.0, 0.5, -0.2], [0.1, 0.2, 0.3]])
    out = layer(x, control_flags=[1, 1])

    assert out.shape == (2, 2)
    audit = layer.audit()
    assert len(audit) == 2


def test_logits_gate_pruning():
    """NetelproLogitsGate sets non-conforming tokens to -inf (0 probability)."""
    gate = NetelproLogitsGate()

    # Vocabulary of 5 tokens: [0, 1, 2, 3, 4]
    # Current rule allows only IDs in range [1, 3] with safety_state=1
    logits = torch.tensor([2.0, 1.5, 3.0, 0.5, 4.0])

    masked = gate.filter_logits(
        logits=logits,
        allowed_min=1,
        allowed_max=3,
        safety_state=1,
    )

    # Tokens 0 and 4 should be -inf
    assert masked[0].item() == float("-inf")
    assert masked[4].item() == float("-inf")
    # Tokens 1, 2, 3 must keep their original logits
    assert masked[1].item() == 1.5
    assert masked[2].item() == 3.0
    assert masked[3].item() == 0.5

    # After softmax, masked tokens must have exact zero probability
    probs = torch.softmax(masked, dim=-1)
    assert probs[0].item() == 0.0
    assert probs[4].item() == 0.0
    assert math.isclose(probs.sum().item(), 1.0, rel_tol=1e-5)


def test_logits_gate_emergency_freeze():
    """When safety_state is 0 (emergency), all actions are pruned."""
    gate = NetelproLogitsGate()
    logits = torch.tensor([1.0, 2.0, 3.0])

    masked = gate.filter_logits(
        logits=logits,
        allowed_min=0,
        allowed_max=5,
        safety_state=0,  # Lockdown
    )

    # All tokens masked to -inf
    assert all(masked[i].item() == float("-inf") for i in range(3))
