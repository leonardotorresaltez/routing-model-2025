

# State of Art for routing problem TSP ( SOTA)

## 1. Pointer Networks (2015)
**Paper:** Vinyals et al., *Pointer Networks*

**Description**
- Encoder–decoder architecture based on **LSTMs**
- Attention mechanism used as a **pointer**
- Outputs sequences of indices (variable-length outputs)

**Pros**
- Conceptually elegant
- Historically important
- Good for learning the core idea

**Cons**
- Slow due to sequential RNN decoding
- Hard to scale
- Not graph-aware
- No longer State of the Art

**My comment**
- Mostly educational today

---

## 2. Graph Pointer Networks (2018–2020)

**Description**
- Encoder replaced by a **Graph Neural Network (GNN)**
- Pointer operates over **graph nodes**
- Commonly trained using **Reinforcement Learning**

**Pros**
- Explicitly models graph structure
- Better than classic Pointer Networks for routing problems

**Cons**
- Decoder is still sequential
- Largely superseded by newer architectures

**My comment**
- Intermediate step, not SOTA anymore

---

## 3. Modern approaches (current SOTA)

### Transformer / Attention Model with masked attention
**Paper:** Kool et al., 2019 – *Attention Model for Routing Problems*

**Description**
- Transformer-style encoder
- Decoder with **masked attention**
- Pointer mechanism is implicit
- Trained with RL (REINFORCE / PPO)

**Pros**
- Faster and more scalable
- Easier to parallelize
- Stable training
- Strong empirical performance
- Widely used as a modern baseline

**My comment** 
- De-facto standard for TSP and routing problems

---

### Graph Transformers
- Combine graph structure with Transformer attention
- Examples: Graphormer, GAT + Transformer

**My comment** 
- Explicit Pointer Networks are often unnecessary — attention handles the selection

---

### Diffusion / Neural Heuristic Models (2022–2024)
- Diffusion models over permutations or routes
- Very strong results on large-scale TSPs
- More complex and research-oriented

---

## Summary

| Approach | Year | Recommended Today | Typical Use |
|--------|------|-------------------|-------------|
| Pointer Networks (LSTM) | 2015 |  No | Learning the concept |
| Graph Pointer Networks | 2018 |  Sometimes | Research |
| Transformer + masked attention | 2019+ | ✅ Yes | Routing / TSP (modern baseline) |
| Graph Transformers | 2021+ | ✅ Yes | State of the Art |
| Diffusion models | 2023+ |  Advanced | Cutting-edge research |

---

## For out project the best is ... ?

For a our course project : *Transformer / Attention Model with masked attention



