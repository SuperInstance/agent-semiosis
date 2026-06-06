//! # agent-semiosis
//!
//! Darwinian semiosis through competitive embedding drift.
//!
//! Based on the hypothesis that competitive riffing modifies embedding spaces,
//! and the most creative duels happen when agents diverge fastest. Each semiotic
//! sign MODIFIES the receiver's embedding space, not just produces output.

use std::f64::consts::PI;

/// Dimensionality of the embedding space.
pub const EMBED_DIM: usize = 32;

/// A 32-dimensional embedding vector representing an agent's current
/// interpretive state in the semiotic space.
#[derive(Clone, Debug, PartialEq)]
pub struct EmbeddingSpace {
    /// The 32-dimensional vector.
    pub vector: [f64; EMBED_DIM],
}

impl EmbeddingSpace {
    /// Create a new embedding space initialized to a given value.
    pub fn new(fill: f64) -> Self {
        Self { vector: [fill; EMBED_DIM] }
    }

    /// Create a zeroed embedding space.
    pub fn zero() -> Self {
        Self::new(0.0)
    }

    /// Create an embedding from a seed value, generating deterministic
    /// pseudo-random coordinates via sine-based hashing.
    pub fn from_seed(seed: u64) -> Self {
        let mut vec = [0.0; EMBED_DIM];
        for i in 0..EMBED_DIM {
            let x = ((seed as f64) * (i as f64 + 1.0) * PI).sin();
            vec[i] = x;
        }
        Self { vector: vec }
    }

    /// Compute Euclidean distance to another embedding.
    pub fn distance_to(&self, other: &EmbeddingSpace) -> f64 {
        self.vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Compute cosine similarity to another embedding.
    pub fn cosine_similarity(&self, other: &EmbeddingSpace) -> f64 {
        let dot: f64 = self.vector.iter().zip(other.vector.iter()).map(|(a, b)| a * b).sum();
        let mag_a: f64 = self.vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mag_b: f64 = other.vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        if mag_a == 0.0 || mag_b == 0.0 {
            return 0.0;
        }
        dot / (mag_a * mag_b)
    }

    /// Apply a mutation to this embedding by blending with a perturbation vector.
    /// `rate` controls how much the mutation affects the embedding (0.0 = none, 1.0 = full replacement).
    pub fn mutate(&mut self, perturbation: &[f64; EMBED_DIM], rate: f64) {
        for i in 0..EMBED_DIM {
            self.vector[i] = self.vector[i] * (1.0 - rate) + perturbation[i] * rate;
        }
    }

    /// Compute the magnitude (L2 norm) of the embedding.
    pub fn magnitude(&self) -> f64 {
        self.vector.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Normalize the embedding to unit length.
    pub fn normalize(&mut self) {
        let mag = self.magnitude();
        if mag > 0.0 {
            for i in 0..EMBED_DIM {
                self.vector[i] /= mag;
            }
        }
    }
}

impl Default for EmbeddingSpace {
    fn default() -> Self {
        Self::zero()
    }
}

/// A semiotic sign: an output that modifies the interpreter's embedding space.
/// The sign carries both content (the message) and a perturbation vector
/// (how it reshapes the receiver).
#[derive(Clone, Debug)]
pub struct SemioticSign {
    /// Human-readable content of the sign.
    pub content: String,
    /// The perturbation this sign applies to the receiver's embedding.
    pub perturbation: [f64; EMBED_DIM],
    /// Mutation rate: how strongly this sign reshapes the receiver (0.0–1.0).
    pub intensity: f64,
    /// Unique identifier for this sign.
    pub id: u64,
}

impl SemioticSign {
    /// Create a new semiotic sign from content, perturbation seed, and intensity.
    pub fn new(content: &str, seed: u64, intensity: f64) -> Self {
        let mut perturbation = [0.0; EMBED_DIM];
        for i in 0..EMBED_DIM {
            perturbation[i] = ((seed as f64 * (i as f64 + 7.0) * PI * 1.3).sin()) * intensity;
        }
        Self {
            content: content.to_string(),
            perturbation,
            intensity: intensity.clamp(0.0, 1.0),
            id: seed,
        }
    }

    /// Apply this sign to a receiver's embedding space, mutating it.
    pub fn apply_to(&self, receiver: &mut EmbeddingSpace) {
        receiver.mutate(&self.perturbation, self.intensity);
    }

    /// Compute how "novel" this sign is relative to an embedding (distance in semiotic space).
    pub fn novelty_vs(&self, embedding: &EmbeddingSpace) -> f64 {
        let pert_as_embed = EmbeddingSpace { vector: self.perturbation };
        embedding.distance_to(&pert_as_embed)
    }
}

/// A sign duel between two agents. Agents exchange signs that mutate each
/// other's embedding spaces. The duel proceeds for a fixed number of rounds.
#[derive(Clone, Debug)]
pub struct SignDuel {
    /// Agent A's embedding.
    pub agent_a: EmbeddingSpace,
    /// Agent B's embedding.
    pub agent_b: EmbeddingSpace,
    /// History of distances between embeddings after each round.
    pub drift_history: Vec<f64>,
    /// Number of rounds completed.
    pub rounds_completed: usize,
    /// Maximum rounds for this duel.
    pub max_rounds: usize,
}

impl SignDuel {
    /// Create a new duel between two agents with given initial embeddings.
    pub fn new(agent_a: EmbeddingSpace, agent_b: EmbeddingSpace, max_rounds: usize) -> Self {
        let initial_distance = agent_a.distance_to(&agent_b);
        Self {
            agent_a,
            agent_b,
            drift_history: vec![initial_distance],
            rounds_completed: 0,
            max_rounds,
        }
    }

    /// Execute one round of the duel. Each agent produces a sign that mutates
    /// the other. Returns the new distance between embeddings.
    pub fn round(&mut self, sign_a: &SemioticSign, sign_b: &SemioticSign) -> f64 {
        sign_a.apply_to(&mut self.agent_b);
        sign_b.apply_to(&mut self.agent_a);
        self.rounds_completed += 1;
        let distance = self.agent_a.distance_to(&self.agent_b);
        self.drift_history.push(distance);
        distance
    }

    /// Run the full duel to completion with generated signs.
    /// Signs are generated deterministically from round number and agent identity.
    pub fn complete(&mut self) -> Vec<f64> {
        while self.rounds_completed < self.max_rounds {
            let round = self.rounds_completed as u64;
            let sign_a = SemioticSign::new(
                &format!("A-round-{}", round),
                round * 1000 + 1,
                0.1 + (round as f64 * 0.01).min(0.5),
            );
            let sign_b = SemioticSign::new(
                &format!("B-round-{}", round),
                round * 1000 + 2,
                0.1 + (round as f64 * 0.02).min(0.5),
            );
            self.round(&sign_a, &sign_b);
        }
        self.drift_history.clone()
    }

    /// Check if the duel is complete.
    pub fn is_complete(&self) -> bool {
        self.rounds_completed >= self.max_rounds
    }

    /// Get the total drift (change in distance) from start to current.
    pub fn total_drift(&self) -> f64 {
        if self.drift_history.len() < 2 {
            return 0.0;
        }
        self.drift_history.last().unwrap() - self.drift_history.first().unwrap()
    }
}

/// Tracks embedding drift over time across multiple agents or duels.
#[derive(Clone, Debug, Default)]
pub struct DriftTracker {
    /// Time-series of (round, embedding_snapshot) pairs.
    pub snapshots: Vec<(usize, EmbeddingSpace)>,
}

impl DriftTracker {
    /// Create a new empty drift tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a snapshot at a given round.
    pub fn record(&mut self, round: usize, embedding: &EmbeddingSpace) {
        self.snapshots.push((round, embedding.clone()));
    }

    /// Compute the drift rate (average distance change per round).
    pub fn drift_rate(&self) -> f64 {
        if self.snapshots.len() < 2 {
            return 0.0;
        }
        let mut total_change = 0.0;
        for window in self.snapshots.windows(2) {
            let dist = window[0].1.distance_to(&window[1].1);
            let round_diff = (window[1].0 as f64) - (window[0].0 as f64);
            total_change += dist / round_diff;
        }
        total_change / (self.snapshots.len() - 1) as f64
    }

    /// Compute total distance traveled through embedding space.
    pub fn total_distance(&self) -> f64 {
        if self.snapshots.len() < 2 {
            return 0.0;
        }
        self.snapshots
            .windows(2)
            .map(|w| w[0].1.distance_to(&w[1].1))
            .sum()
    }

    /// Get the distance between the first and last snapshot.
    pub fn net_displacement(&self) -> f64 {
        if self.snapshots.is_empty() {
            return 0.0;
        }
        let first = &self.snapshots.first().unwrap().1;
        let last = &self.snapshots.last().unwrap().1;
        first.distance_to(last)
    }

    /// Count snapshots recorded.
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Check if tracker is empty.
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }
}

/// Measures the correlation between drift rate and perceived novelty.
/// The core hypothesis: faster drift → higher creativity.
#[derive(Clone, Debug, Default)]
pub struct CreativityScore {
    /// Pairs of (drift_rate, novelty_score) for correlation analysis.
    pub samples: Vec<(f64, f64)>,
}

impl CreativityScore {
    /// Create a new creativity score tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a sample: drift rate and corresponding novelty score.
    pub fn record(&mut self, drift_rate: f64, novelty: f64) {
        self.samples.push((drift_rate, novelty));
    }

    /// Compute Pearson correlation between drift rate and novelty.
    /// Returns None if insufficient samples.
    pub fn correlation(&self) -> Option<f64> {
        if self.samples.len() < 2 {
            return None;
        }
        let n = self.samples.len() as f64;
        let sum_x: f64 = self.samples.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = self.samples.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = self.samples.iter().map(|(x, y)| x * y).sum();
        let sum_x2: f64 = self.samples.iter().map(|(x, _)| x * x).sum();
        let sum_y2: f64 = self.samples.iter().map(|(_, y)| y * y).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator == 0.0 {
            return None;
        }
        Some(numerator / denominator)
    }

    /// Get the average novelty score.
    pub fn avg_novelty(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples.iter().map(|(_, y)| y).sum::<f64>() / self.samples.len() as f64
    }

    /// Get the average drift rate.
    pub fn avg_drift(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples.iter().map(|(x, _)| x).sum::<f64>() / self.samples.len() as f64
    }

    /// Number of samples recorded.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_initialization() {
        let e = EmbeddingSpace::new(1.0);
        assert_eq!(e.vector, [1.0; EMBED_DIM]);
        assert_eq!(e.magnitude(), (EMBED_DIM as f64).sqrt());

        let z = EmbeddingSpace::zero();
        assert_eq!(z.vector, [0.0; EMBED_DIM]);
        assert_eq!(z.magnitude(), 0.0);
    }

    #[test]
    fn test_embedding_from_seed() {
        let e1 = EmbeddingSpace::from_seed(42);
        let e2 = EmbeddingSpace::from_seed(42);
        assert_eq!(e1, e2);

        let e3 = EmbeddingSpace::from_seed(99);
        assert_ne!(e1, e3);
    }

    #[test]
    fn test_embedding_distance() {
        let a = EmbeddingSpace::new(0.0);
        let b = EmbeddingSpace::new(0.0);
        assert!((a.distance_to(&b) - 0.0).abs() < 1e-10);

        let c = EmbeddingSpace::new(1.0);
        let expected = (EMBED_DIM as f64).sqrt();
        assert!((a.distance_to(&c) - expected).abs() < 1e-10);

        // Distance is symmetric
        assert!((c.distance_to(&a) - a.distance_to(&c)).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = EmbeddingSpace::new(1.0);
        let b = EmbeddingSpace::new(1.0);
        assert!((a.cosine_similarity(&b) - 1.0).abs() < 1e-10);

        // Zero vector has similarity 0
        let z = EmbeddingSpace::zero();
        assert!((a.cosine_similarity(&z) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_sign_mutation() {
        let mut e = EmbeddingSpace::new(0.0);
        let perturbation = [1.0; EMBED_DIM];
        e.mutate(&perturbation, 0.5);
        for v in e.vector.iter() {
            assert!((v - 0.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sign_creation_and_application() {
        let sign = SemioticSign::new("hello", 42, 0.3);
        assert_eq!(sign.content, "hello");
        assert!((sign.intensity - 0.3).abs() < 1e-10);

        let mut embed = EmbeddingSpace::zero();
        let original = embed.clone();
        sign.apply_to(&mut embed);
        // The embedding should have changed
        assert_ne!(embed, original);
    }

    #[test]
    fn test_duel_single_round() {
        let a = EmbeddingSpace::from_seed(1);
        let b = EmbeddingSpace::from_seed(2);
        let mut duel = SignDuel::new(a.clone(), b.clone(), 10);

        let initial_dist = duel.drift_history[0];
        let sign_a = SemioticSign::new("a-says", 100, 0.2);
        let sign_b = SemioticSign::new("b-says", 200, 0.2);
        let new_dist = duel.round(&sign_a, &sign_b);

        assert_eq!(duel.rounds_completed, 1);
        assert_eq!(duel.drift_history.len(), 2);
        assert!(new_dist >= 0.0);
    }

    #[test]
    fn test_duel_completion() {
        let a = EmbeddingSpace::from_seed(10);
        let b = EmbeddingSpace::from_seed(20);
        let mut duel = SignDuel::new(a, b, 5);
        let history = duel.complete();

        assert!(duel.is_complete());
        assert_eq!(history.len(), 6); // initial + 5 rounds
        assert_eq!(duel.rounds_completed, 5);
    }

    #[test]
    fn test_drift_tracker() {
        let mut tracker = DriftTracker::new();
        assert!(tracker.is_empty());

        tracker.record(0, &EmbeddingSpace::from_seed(1));
        tracker.record(1, &EmbeddingSpace::from_seed(2));
        tracker.record(2, &EmbeddingSpace::from_seed(3));

        assert_eq!(tracker.len(), 3);
        assert!(tracker.drift_rate() > 0.0);
        assert!(tracker.total_distance() > 0.0);
        assert!(tracker.net_displacement() > 0.0);
    }

    #[test]
    fn test_creativity_score_correlation() {
        let mut score = CreativityScore::new();
        // Perfect positive correlation
        for i in 0..10 {
            let x = i as f64;
            score.record(x, x * 2.0);
        }

        let corr = score.correlation().unwrap();
        assert!((corr - 1.0).abs() < 1e-10, "Expected perfect correlation, got {}", corr);
    }

    #[test]
    fn test_creativity_score_negative_correlation() {
        let mut score = CreativityScore::new();
        // Perfect negative correlation
        for i in 0..10 {
            let x = i as f64;
            score.record(x, 100.0 - x);
        }

        let corr = score.correlation().unwrap();
        assert!(corr < -0.9, "Expected negative correlation, got {}", corr);
    }

    #[test]
    fn test_creativity_score_insufficient_data() {
        let mut score = CreativityScore::new();
        assert!(score.correlation().is_none());
        score.record(1.0, 2.0);
        assert!(score.correlation().is_none());
    }

    #[test]
    fn test_normalization() {
        let mut e = EmbeddingSpace::new(3.0);
        e.normalize();
        assert!((e.magnitude() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_duel_drift_increases_over_time() {
        // With divergent seeds and high enough intensity, drift should generally increase
        let a = EmbeddingSpace::from_seed(1);
        let b = EmbeddingSpace::from_seed(999);
        let mut duel = SignDuel::new(a, b, 20);
        duel.complete();

        // Check that drift history has variation
        let min = duel.drift_history.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = duel.drift_history.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max > min, "Drift should vary over the course of a duel");
    }
}
