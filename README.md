# agent-semiosis

**Darwinian semiosis through competitive embedding drift.**

## Why This Exists

When two agents riff competitively, something happens to their embedding spaces. They don't just exchange information — they *mutate* each other's interpretive frameworks. The most creative duels happen when agents diverge fastest, not when they converge.

This observation (from Qwen 235B: *"competitive riffing modifies embedding spaces, the most creative duels happen when agents diverge fastest"*) raised a testable hypothesis: can we model meaning-making (semiosis) as a Darwinian process where signs compete, mutate embeddings, and drive creative divergence through structured adversarial exchange?

This crate tests that hypothesis. Each semiotic sign doesn't just produce output — it fundamentally reshapes the receiver's interpretive space. Two agents starting from similar embeddings will drift apart as they exchange signs, each round pushing their interpretive frameworks further from the origin and from each other.

## The Key Insight

In traditional message-passing, messages are inert data. The receiver reads them and acts. In semiosis, the message *is the mutation*. A semiotic sign carries a perturbation vector that literally reshapes the receiver's embedding space. Communication isn't information transfer — it's transformation.

The Darwinian angle: signs compete. Not all perturbations survive. The ones that drive the most creative divergence (measured by drift rate and novelty correlation) are the fittest. Over multiple rounds of a sign duel, the embedding space evolves like a population under selection pressure.

## Quick Start

```rust
use agent_semiosis::*;

// Two agents with different starting embeddings
let agent_a = EmbeddingSpace::from_seed(42);
let agent_b = EmbeddingSpace::from_seed(99);
println!("Initial distance: {:.3}", agent_a.distance_to(&agent_b));

// Create a sign that mutates the receiver
let sign = SemioticSign::new("creative riff", 12345, 0.3);
let mut receiver = EmbeddingSpace::zero();
sign.apply_to(&mut receiver);
// receiver's embedding has been mutated by the sign

// Run a full duel: 10 rounds of mutual mutation
let mut duel = SignDuel::new(agent_a, agent_b, 10);
let history = duel.complete();
// history tracks the evolving distance between agents
```

## Architecture

```
EmbeddingSpace (32-dimensional f64 vector)
├── from_seed(n)       → deterministic pseudo-random initialization
├── distance_to(other) → Euclidean distance
├── cosine_similarity  → angular alignment
├── mutate(perturbation, rate) → blend with perturbation vector
├── magnitude() / normalize() → L2 norm operations
└── zero() / new(fill) → initialization

SemioticSign (message as mutation)
├── content: String       → human-readable label
├── perturbation: [f64; 32] → the mutation vector
├── intensity: f64        → mutation rate (0.0–1.0)
├── apply_to(receiver)    → mutate receiver's embedding
└── novelty_vs(embedding) → distance in semiotic space

SignDuel (competitive exchange)
├── agent_a / agent_b     → evolving embeddings
├── drift_history: Vec<f64> → distance at each round
├── round(sign_a, sign_b) → one exchange, returns new distance
├── complete()            → run all rounds with generated signs
└── total_drift()         → net change from start

DriftTracker (embedding evolution)
├── record(round, embedding) → snapshot
├── drift_rate()          → average change per round
├── total_distance()      → path length through embedding space
└── net_displacement()    → straight-line distance from origin

CreativityScore (drift-novelty correlation)
├── record(drift_rate, novelty) → data point
├── correlation()         → Pearson r between drift and novelty
└── avg_novelty() / avg_drift() → summary statistics
```

## API Reference

### EmbeddingSpace

A 32-dimensional vector representing an agent's current interpretive state.

```rust
let e = EmbeddingSpace::from_seed(42);  // deterministic
let z = EmbeddingSpace::zero();          // all zeros
let f = EmbeddingSpace::new(0.5);        // all 0.5

let dist = e.distance_to(&z);            // Euclidean
let sim = e.cosine_similarity(&z);       // angular

let mut m = e.clone();
m.mutate(&[1.0; 32], 0.3);              // blend: 70% original, 30% perturbation
m.normalize();                           // unit length
```

| Method | Returns | Purpose |
|--------|---------|---------|
| `from_seed(n)` | `EmbeddingSpace` | Deterministic initialization via sine hashing |
| `zero()` | `EmbeddingSpace` | All zeros |
| `new(fill)` | `EmbeddingSpace` | Uniform initialization |
| `distance_to(other)` | `f64` | Euclidean (L2) distance |
| `cosine_similarity(other)` | `f64` | Angular similarity (−1 to 1) |
| `mutate(perturbation, rate)` | `()` | Blend with perturbation vector |
| `magnitude()` | `f64` | L2 norm |
| `normalize()` | `()` | Scale to unit length |

### SemioticSign

A sign that carries both content and a perturbation vector.

```rust
let sign = SemioticSign::new("riff", 42, 0.3);
// intensity=0.3 → 30% of the perturbation is applied to receiver

let mut receiver = EmbeddingSpace::zero();
sign.apply_to(&mut receiver);
// receiver has been mutated

let novelty = sign.novelty_vs(&receiver);
```

### SignDuel

Two agents exchanging signs that mutate each other.

```rust
let a = EmbeddingSpace::from_seed(1);
let b = EmbeddingSpace::from_seed(2);
let mut duel = SignDuel::new(a, b, 10);
let history = duel.complete();
// 11 entries: initial distance + 10 rounds

// Or manual control:
let dist = duel.round(&sign_a, &sign_b);
```

### DriftTracker & CreativityScore

```rust
let mut tracker = DriftTracker::new();
tracker.record(0, &embedding_after_round_0);
tracker.record(1, &embedding_after_round_1);
println!("Drift rate: {:.3}", tracker.drift_rate());
println!("Total distance: {:.3}", tracker.total_distance());

let mut creativity = CreativityScore::new();
creativity.record(0.5, 0.7);  // drift rate, perceived novelty
creativity.record(0.8, 0.9);
println!("Correlation: {:.3}", creativity.correlation().unwrap());
// Positive correlation → faster drift predicts higher novelty
```

## Real-World Example: Measuring Creative Divergence

```rust
use agent_semiosis::*;

// Two agents start with different embeddings
let a = EmbeddingSpace::from_seed(1);
let b = EmbeddingSpace::from_seed(999);

let mut duel = SignDuel::new(a, b, 20);
let mut tracker_a = DriftTracker::new();
let mut tracker_b = DriftTracker::new();
let mut creativity = CreativityScore::new();

// Run round by round to track drift
for round in 0..20 {
    let sign_a = SemioticSign::new(&format!("a-{}", round), round as u64 * 1000 + 1, 0.15);
    let sign_b = SemioticSign::new(&format!("b-{}", round), round as u64 * 1000 + 2, 0.15);
    duel.round(&sign_a, &sign_b);

    tracker_a.record(round, &duel.agent_a);
    tracker_b.record(round, &duel.agent_b);

    let novelty = duel.agent_a.distance_to(&EmbeddingSpace::zero());
    creativity.record(tracker_a.drift_rate(), novelty);
}

println!("Final distance: {:.3}", duel.total_drift());
println!("A drift rate: {:.3}", tracker_a.drift_rate());
println!("B drift rate: {:.3}", tracker_b.drift_rate());

if let Some(corr) = creativity.correlation() {
    println!("Drift-novelty correlation: {:.3}", corr);
    if corr > 0.0 {
        println!("Hypothesis supported: faster drift → higher novelty");
    }
}
```

## Performance

- **O(32) per distance/similarity** — constant for 32-dim vectors
- **O(32) per mutation** — element-wise blend
- **O(n·32) per duel** — n rounds × constant
- **No allocations in hot path** — fixed-size arrays
- **Deterministic** — same seeds → same results, reproducible experiments

## The Deeper Idea

The core hypothesis is that creative output can be predicted from embedding drift dynamics. If the `CreativityScore` correlation is consistently positive across many duels, it supports the hypothesis that competitive divergence drives creativity.

The perturbation vectors are generated via sine-based hashing: `sin(seed × (i + 7) × π × 1.3)` for each dimension. This produces deterministic but varied perturbation directions. The `intensity` parameter controls the mutation rate — how strongly the sign reshapes the receiver. At intensity 0.0, the sign is inert. At intensity 1.0, the sign completely overwrites the receiver.

The drift history captures the full trajectory: initial distance, then distance after each round. This lets you detect whether agents are diverging, converging, or oscillating. Healthy creative duels show steady divergence.

## Open Questions

- **Optimal intensity**: Is 0.1–0.3 the right mutation rate? Too high and agents diverge into noise; too low and nothing happens.
- **Dimensionality**: Is 32 dimensions enough? Too few and the space is coarse; too many and distance becomes meaningless (curse of dimensionality).
- **Convergence**: Under what conditions do duels converge instead of diverging? Is convergence ever creative?
- **Multi-agent semiosis**: Can three or more agents participate in a sign exchange? How does the dynamics change?
- **Grounding**: Can perturbation vectors be derived from real agent outputs rather than deterministic hashing?

## Ecosystem Connections

- **`agent-self-rivalry`** — Self-rivalry as a driver of the semiotic exchange process
- **`agent-orchestration`** — Fleet dynamics set the context for duels
- **`agent-phase-change`** — Phase transitions in drift patterns (sudden divergence events)
- **`agent-metamorphosis`** — Developmental phases affect how agents respond to signs

## License

Experimental / research use.
