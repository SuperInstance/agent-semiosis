# agent-semiosis

**Darwinian semiosis through competitive embedding drift.**

An experimental Rust library exploring the hypothesis that competitive interaction between agents modifies their embedding spaces, and that the most creative outcomes emerge when agents diverge fastest. Each semiotic sign doesn't just produce output — it fundamentally reshapes the receiver's interpretive space.

## Origin

Based on Qwen's insight: *"competitive riffing modifies embedding spaces, the most creative duels happen when agents diverge fastest"*

This crate tests whether meaning-making (semiosis) can be modeled as a Darwinian process where signs compete, mutate embeddings, and drive creative divergence through structured adversarial exchange.

## Core Concepts

### EmbeddingSpace
A 32-dimensional `f64` vector representing an agent's current interpretive state. Not a static representation — it evolves through every interaction. Two agents starting from similar embeddings will drift apart as they exchange signs, each round pushing their interpretive frameworks further from the origin and from each other.

```rust
use agent_semiosis::EmbeddingSpace;

let agent_a = EmbeddingSpace::from_seed(42);
let agent_b = EmbeddingSpace::from_seed(99);
let distance = agent_a.distance_to(&agent_b);
```

### SemioticSign
A sign is an output that modifies the interpreter's embedding space. Unlike traditional message-passing where messages are inert data, a semiotic sign carries a perturbation vector that literally reshapes the receiver. The `intensity` field controls mutation rate — higher intensity means the sign more aggressively overwrites the receiver's embedding.

```rust
use agent_semiosis::{SemioticSign, EmbeddingSpace};

let sign = SemioticSign::new("creative riff", 12345, 0.3);
let mut receiver = EmbeddingSpace::zero();
sign.apply_to(&mut receiver);
// receiver's embedding has been mutated
```

### SignDuel
The central abstraction: two agents exchange signs that mutate each other's embeddings over multiple rounds. Each round produces a new distance between the agents' embeddings, building a drift history that captures the trajectory of their divergence (or convergence).

```rust
use agent_semiosis::{EmbeddingSpace, SignDuel};

let a = EmbeddingSpace::from_seed(1);
let b = EmbeddingSpace::from_seed(2);
let mut duel = SignDuel::new(a, b, 10);
let history = duel.complete();
// history tracks the evolving distance between agents
```

### DriftTracker
Records snapshots of an embedding over time and computes drift metrics: drift rate (average change per round), total distance traveled, and net displacement. This answers the question: "how much has this agent's interpretive space actually changed?"

### CreativityScore
Measures the Pearson correlation between drift rate and perceived novelty. The hypothesis predicts a positive correlation: agents whose embeddings change faster should be perceived as more creative. This is the experimental test of the core claim.

## Design Principles

1. **Signs mutate, not just inform.** Every `SemioticSign` carries a perturbation vector that reshapes the receiver's embedding space. Communication is transformation.

2. **Divergence is creativity.** The drift history tracks how far apart agents grow. The crate provides tools to measure whether faster divergence correlates with higher novelty scores.

3. **Embeddings are living things.** An `EmbeddingSpace` is not a fixed representation. It's a dynamic entity that evolves through every interaction, capturing the agent's current interpretive stance.

## Metrics

- **Euclidean distance**: Standard L2 distance between embeddings
- **Cosine similarity**: Angular similarity, useful for comparing direction of drift
- **Drift rate**: Average change in embedding per round
- **Pearson correlation**: Statistical measure of drift-novelty relationship
- **Net displacement**: How far the embedding has moved from its origin

## Testing

The crate includes 14 comprehensive tests covering:
- Embedding initialization (zero, seed-based, custom fill)
- Distance and similarity calculations
- Sign creation and mutation mechanics
- Single-round and multi-round duels
- Drift tracking and rate computation
- Creativity correlation (positive and negative)
- Normalization and edge cases

Run tests with:
```bash
cargo test
```

## Experimental Hypothesis

The crate is designed to test whether creative output can be predicted from embedding drift dynamics. If the `CreativityScore` correlation is consistently positive across many duels, it supports the hypothesis that competitive divergence drives creativity. If not, the relationship may be more nuanced than simple drift-novelty correlation.

## License

Experimental / research use.
