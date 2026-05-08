# RLT Paper Reference

**Title**: Precise Manipulation with Efficient Online RL  
**Authors**: Charles Xu, Jost Tobias Springenberg, Michael Equi, Ali Amin, Adnan Esmail, Sergey Levine, Liyiming Ke  
**Organization**: Physical Intelligence (π)  
**Published**: March 19, 2026  
**URL**: https://www.pi.website/research/rlt  
**PDF**: Available at the URL above

---

## Key Ideas

### 1. RL Token — The Core Contribution

> "We train the VLA to produce an 'RL token' that summarizes the VLA's internal
> representations. This RL token is then used as the input into a much smaller
> model that can be trained with RL in real time."

- An encoder-decoder transformer is added to the VLA
- The encoder compresses all VLM prefix tokens (images + language) through a bottleneck
- The single output token (RL token) retains enough information for reconstruction
- After training, only the encoder is kept; the decoder is discarded
- The RL token provides a compact state representation for actor/critic

### 2. Actor Design Decisions

1. **Action chunking**: RL policy predicts action chunks (matching VLA output structure)
2. **VLA reference conditioning**: Actor receives VLA's predicted action as input — learns to *edit* rather than replace
3. **BC regularization**: Policy is regularized toward the VLA reference action
4. **Reference-action dropout**: 50% of training batches zero out the reference to prevent copying
5. **Human interventions**: Corrections can be folded back into training

### 3. Training Process

1. Fine-tune VLA on task demonstrations (offline, once)
2. Train RL token encoder-decoder on VLA embeddings (offline, once)
3. Online RL training with small actor + critic using RL token as state
4. As little as 15 minutes of real-world data for improvement

### 4. Results (from paper)

| Task | Base VLA → After RLT | Training Time |
|------|----------------------|---------------|
| Screw insertion (M3) | Slow → 3× faster | ~hours |
| Zip tie fastening | Low success → high success | ~hours |
| Ethernet cable insertion | Moderate → 3× throughput | 15 min robot data |
| Power cord insertion | Moderate → 3× throughput | ~hours |

### 5. Architecture Details

```
VLA (π0.5/π0.6) backbone:
  - PaliGemma (SigLIP + Gemma-2B) → prefix tokens (N ≈ 527)
  - Gemma 300M action expert → action chunks (H=50 steps)

RL Token:
  - 4-layer transformer encoder with <rl> query
  - 4-layer transformer decoder (training only)
  - Bottleneck: 2048-d → 512-d (or task-dependent)

Actor/Critic:
  - Small MLPs: 2-layer, hidden=256 (or 3-layer, hidden=512 for harder tasks)
  - Input: z_rl(512) + proprio + reference_chunk
  - Output: action chunk (C=10 steps × action_dim)
  - UTD=5 (updates per environment step)
```

### 6. Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| RL chunk C | 10 | Steps per RL action |
| VLA chunk H | 50 | Steps per VLA action |
| BC weight β | ~1.0 | Regularization strength |
| Reference dropout | 50% | Prevents mode collapse |
| Update ratio G | 5 | Lower than standard RLPD |
| Encoder layers | 4 | Transformer encoder |
| Decoder layers | 4 | Transformer decoder |
| Token dim | 512-2048 | Information bottleneck |

---

## Citation

```bibtex
@article{xu2026rlt,
  title={Precise Manipulation with Efficient Online RL},
  author={Xu, Charles and Springenberg, Jost Tobias and Equi, Michael and Amin, Ali and Esmail, Adnan and Levine, Sergey and Ke, Liyiming},
  journal={Physical Intelligence Research},
  year={2026},
  url={https://www.pi.website/research/rlt}
}
```
