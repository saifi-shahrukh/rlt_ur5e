"""
Unit tests for the RL Token encoder-decoder model.

Run with:
    cd ~/ur5e_hande_workspace/rlt_ur5e
    python -m pytest rlt/tests/test_rl_token.py -v

Or standalone:
    python rlt/tests/test_rl_token.py
"""
import sys
from pathlib import Path

import torch
import numpy as np

# Ensure rlt is importable
sys.path.insert(0, str(Path(__file__).parents[2]))

from rlt.models.rl_token import RLTokenModel


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Model Instantiation
# ═══════════════════════════════════════════════════════════════════════════════


def test_model_creation():
    """Test that model can be created with default params."""
    model = RLTokenModel(embed_dim=2048, token_dim=512)
    assert model is not None
    params = model.get_num_params()
    assert params["total"] > 0
    print(f"  Model created: {params['total']:,} total params")
    print(f"    Encoder: {params['encoder']:,}")
    print(f"    Decoder: {params['decoder']:,}")


def test_model_creation_small():
    """Test model with small dimensions (fast)."""
    model = RLTokenModel(
        embed_dim=128, token_dim=32, enc_layers=2, dec_layers=2,
        n_heads=4, ffn_dim=256, max_len=100
    )
    params = model.get_num_params()
    assert params["total"] > 0
    assert params["total"] < 5_000_000  # Should be small
    print(f"  Small model: {params['total']:,} params")


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Encoder
# ═══════════════════════════════════════════════════════════════════════════════


def test_encode_shape():
    """Test encoder output shape."""
    model = RLTokenModel(embed_dim=256, token_dim=64, enc_layers=2,
                         dec_layers=2, n_heads=4, ffn_dim=512)
    z = torch.randn(2, 50, 256)  # B=2, N=50, D=256
    z_rl = model.encode(z)
    assert z_rl.shape == (2, 64), f"Expected (2, 64), got {z_rl.shape}"
    print(f"  Encoder: (2, 50, 256) → {z_rl.shape}")


def test_encode_different_seq_lengths():
    """Test encoder works with different sequence lengths."""
    model = RLTokenModel(embed_dim=128, token_dim=32, enc_layers=2,
                         dec_layers=2, n_heads=4, ffn_dim=256)

    for N in [10, 50, 100, 527]:
        z = torch.randn(1, N, 128)
        z_rl = model.encode(z)
        assert z_rl.shape == (1, 32), f"Failed for N={N}: got {z_rl.shape}"

    print(f"  Encoder works for N ∈ [10, 50, 100, 527]")


def test_encode_batch_independence():
    """Test that encoding is independent per batch element."""
    model = RLTokenModel(embed_dim=128, token_dim=32, enc_layers=2,
                         dec_layers=2, n_heads=4, ffn_dim=256)
    model.eval()

    z1 = torch.randn(1, 30, 128)
    z2 = torch.randn(1, 30, 128)
    z_batch = torch.cat([z1, z2], dim=0)

    with torch.no_grad():
        z_rl_1 = model.encode(z1)
        z_rl_2 = model.encode(z2)
        z_rl_batch = model.encode(z_batch)

    assert torch.allclose(z_rl_batch[0], z_rl_1[0], atol=1e-5)
    assert torch.allclose(z_rl_batch[1], z_rl_2[0], atol=1e-5)
    print(f"  Batch independence verified")


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Decoder
# ═══════════════════════════════════════════════════════════════════════════════


def test_decode_shape():
    """Test decoder output shape matches input."""
    model = RLTokenModel(embed_dim=256, token_dim=64, enc_layers=2,
                         dec_layers=2, n_heads=4, ffn_dim=512)
    z_rl = torch.randn(2, 64)
    z_tgt = torch.randn(2, 50, 256)
    z_hat = model.decode_tf(z_rl, z_tgt)
    assert z_hat.shape == (2, 50, 256), f"Expected (2, 50, 256), got {z_hat.shape}"
    print(f"  Decoder: z_rl(2,64) + tgt(2,50,256) → {z_hat.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Full Training Forward Pass
# ═══════════════════════════════════════════════════════════════════════════════


def test_compute_loss():
    """Test full forward pass returns valid loss and z_rl."""
    model = RLTokenModel(embed_dim=256, token_dim=64, enc_layers=2,
                         dec_layers=2, n_heads=4, ffn_dim=512)
    z = torch.randn(4, 50, 256)
    loss, z_rl = model.compute_loss(z)

    assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
    assert not torch.isnan(loss), "Loss is NaN!"
    assert not torch.isinf(loss), "Loss is Inf!"
    assert loss.item() > 0, "Loss should be positive"
    assert z_rl.shape == (4, 64)
    print(f"  Loss: {loss.item():.5f} (should be > 0)")
    print(f"  z_rl shape: {z_rl.shape}")


def test_loss_requires_grad():
    """Test that loss has gradient (can backprop)."""
    model = RLTokenModel(embed_dim=128, token_dim=32, enc_layers=2,
                         dec_layers=2, n_heads=4, ffn_dim=256)
    z = torch.randn(2, 30, 128)
    loss, _ = model.compute_loss(z)

    assert loss.requires_grad, "Loss should require grad"
    loss.backward()

    # Check gradients flow to encoder
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "No gradients flowing!"
    print(f"  Gradients verified (loss={loss.item():.5f})")


def test_loss_decreases_with_training():
    """Test that loss actually decreases with gradient steps."""
    torch.manual_seed(42)
    model = RLTokenModel(embed_dim=128, token_dim=32, enc_layers=2,
                         dec_layers=2, n_heads=4, ffn_dim=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Use same batch for overfitting test
    z = torch.randn(8, 30, 128)

    losses = []
    for step in range(100):
        loss, _ = model.compute_loss(z)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Loss should decrease significantly
    initial_loss = np.mean(losses[:5])
    final_loss = np.mean(losses[-5:])
    improvement = (initial_loss - final_loss) / initial_loss

    assert final_loss < initial_loss, \
        f"Loss didn't decrease: {initial_loss:.4f} → {final_loss:.4f}"
    assert improvement > 0.1, \
        f"Loss decreased only {improvement*100:.1f}% (need >10%)"

    print(f"  Loss: {initial_loss:.4f} → {final_loss:.4f} "
          f"({improvement*100:.1f}% improvement in 100 steps)")


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Inference Mode
# ═══════════════════════════════════════════════════════════════════════════════


def test_extract_no_grad():
    """Test that extract() works without gradients."""
    model = RLTokenModel(embed_dim=256, token_dim=64, enc_layers=2,
                         dec_layers=2, n_heads=4, ffn_dim=512)
    model.eval()

    z = torch.randn(1, 527, 256)
    z_rl = model.extract(z)

    assert z_rl.shape == (1, 64)
    assert not z_rl.requires_grad
    print(f"  Extract (inference): (1, 527, 256) → {z_rl.shape}")


def test_extract_deterministic():
    """Test that extract() is deterministic."""
    model = RLTokenModel(embed_dim=128, token_dim=32, enc_layers=2,
                         dec_layers=2, n_heads=4, ffn_dim=256)
    model.eval()

    z = torch.randn(1, 50, 128)
    z_rl_1 = model.extract(z)
    z_rl_2 = model.extract(z)

    assert torch.allclose(z_rl_1, z_rl_2, atol=1e-6)
    print(f"  Extract is deterministic ✓")


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Realistic Dimensions (Gemma-2B)
# ═══════════════════════════════════════════════════════════════════════════════


def test_realistic_dimensions():
    """Test with actual Gemma-2B dimensions (embed=2048, N≈527)."""
    model = RLTokenModel(
        embed_dim=2048,
        token_dim=512,
        enc_layers=4,
        dec_layers=4,
        n_heads=8,
        ffn_dim=2048,
    )

    # Simulate real VLM output: 2 cameras × 256 patches + 15 lang tokens = 527
    z = torch.randn(1, 527, 2048)
    loss, z_rl = model.compute_loss(z)

    assert z_rl.shape == (1, 512)
    assert not torch.isnan(loss)
    params = model.get_num_params()

    print(f"  Realistic test: (1, 527, 2048) → z_rl (1, 512)")
    print(f"  Loss: {loss.item():.5f}")
    print(f"  Total params: {params['total']:,}")
    print(f"  Memory: ~{params['total'] * 4 / 1024 / 1024:.1f} MB (float32)")


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Buffer Integration (smoke test)
# ═══════════════════════════════════════════════════════════════════════════════


def test_buffer_integration():
    """Test that RL token output works with RLT buffer."""
    from rlt.agents.rlt_buffer import RLTBuffer

    model = RLTokenModel(embed_dim=128, token_dim=32, enc_layers=2,
                         dec_layers=2, n_heads=4, ffn_dim=256)
    model.eval()

    buffer = RLTBuffer(
        capacity=100, token_dim=32, proprio_dim=7,
        action_dim=6, chunk_size=10
    )

    # Simulate an episode
    T = 30
    z_rl_list = []
    for t in range(T):
        z = torch.randn(1, 50, 128)
        z_rl = model.extract(z).squeeze(0).numpy()  # (32,)
        z_rl_list.append(z_rl)

    proprio_list = [np.random.randn(7).astype(np.float32) for _ in range(T)]
    action_list = [np.random.randn(6).astype(np.float32) for _ in range(T)]
    ref_list = [np.random.randn(6).astype(np.float32) for _ in range(T)]

    added = buffer.add_episode(
        z_rl_list=z_rl_list,
        proprio_list=proprio_list,
        action_list=action_list,
        ref_list=ref_list,
        reward=1.0,
        done=True,
    )

    assert added > 0, "No transitions added!"
    assert len(buffer) == added

    # Sample from buffer
    batch = buffer.sample(min(4, added))
    assert batch["z_rl"].shape[1] == 32
    assert batch["action_chunk"].shape[1:] == (10, 6)

    print(f"  Buffer integration: {added} transitions added, sample works ✓")


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Checkpoint Save/Load
# ═══════════════════════════════════════════════════════════════════════════════


def test_checkpoint_save_load(tmp_path=None):
    """Test saving and loading model checkpoint."""
    import tempfile

    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())

    ckpt_path = tmp_path / "test_rl_token.pt"

    # Train briefly
    model = RLTokenModel(embed_dim=128, token_dim=32, enc_layers=2,
                         dec_layers=2, n_heads=4, ffn_dim=256)
    z = torch.randn(4, 30, 128)
    loss, _ = model.compute_loss(z)

    # Save (include ALL config params needed to reconstruct)
    torch.save({
        "model": model.state_dict(),
        "config": {
            "embed_dim": 128, "token_dim": 32,
            "enc_layers": 2, "dec_layers": 2,
            "n_heads": 4, "ffn_dim": 256,
        },
        "loss": loss.item(),
        "step": 100,
    }, ckpt_path)

    # Load
    ckpt = torch.load(ckpt_path, weights_only=False)
    model2 = RLTokenModel(
        embed_dim=ckpt["config"]["embed_dim"],
        token_dim=ckpt["config"]["token_dim"],
        enc_layers=ckpt["config"]["enc_layers"],
        dec_layers=ckpt["config"]["dec_layers"],
        n_heads=ckpt["config"]["n_heads"],
        ffn_dim=ckpt["config"]["ffn_dim"],
    )
    model2.load_state_dict(ckpt["model"])
    model2.eval()

    # Verify outputs match
    model.eval()
    z_test = torch.randn(1, 30, 128)
    z_rl_1 = model.extract(z_test)
    z_rl_2 = model2.extract(z_test)
    assert torch.allclose(z_rl_1, z_rl_2, atol=1e-6)

    print(f"  Save/Load verified: outputs match ✓")
    # Cleanup
    ckpt_path.unlink()


# ═══════════════════════════════════════════════════════════════════════════════
# Main: Run all tests
# ═══════════════════════════════════════════════════════════════════════════════


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        ("Model creation (default)", test_model_creation),
        ("Model creation (small)", test_model_creation_small),
        ("Encoder shape", test_encode_shape),
        ("Encoder variable lengths", test_encode_different_seq_lengths),
        ("Encoder batch independence", test_encode_batch_independence),
        ("Decoder shape", test_decode_shape),
        ("Compute loss", test_compute_loss),
        ("Loss requires grad", test_loss_requires_grad),
        ("Loss decreases", test_loss_decreases_with_training),
        ("Extract (no grad)", test_extract_no_grad),
        ("Extract deterministic", test_extract_deterministic),
        ("Realistic dimensions (2048-d)", test_realistic_dimensions),
        ("Buffer integration", test_buffer_integration),
        ("Checkpoint save/load", test_checkpoint_save_load),
    ]

    print("\n" + "=" * 60)
    print("  RL Token Model — Unit Tests")
    print("=" * 60 + "\n")

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests:
        try:
            print(f"[TEST] {name}")
            test_fn()
            passed += 1
            print(f"  ✅ PASSED\n")
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"  ❌ FAILED: {e}\n")

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    if errors:
        print("\nFailures:")
        for name, err in errors:
            print(f"  • {name}: {err}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
