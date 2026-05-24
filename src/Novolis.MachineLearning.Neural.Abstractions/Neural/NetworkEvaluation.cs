namespace Novolis.MachineLearning.Neural;

/// <summary>Forward-pass diagnostics for analysis and training.</summary>
/// <param name="Output">Final layer output values.</param>
/// <param name="Activations">Per-layer activation values.</param>
/// <param name="WeightedSums">Per-layer pre-activation sums.</param>
public sealed record NetworkEvaluation(
    double[] Output,
    double[][] Activations,
    double[][] WeightedSums);
