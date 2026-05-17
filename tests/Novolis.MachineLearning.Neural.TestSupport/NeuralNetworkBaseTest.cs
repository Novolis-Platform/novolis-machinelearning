using Novolis.MachineLearning.Neural;
using Novolis.MachineLearning.Neural.Kit;
using Novolis.MachineLearning.TestInfrastructure;

namespace Novolis.MachineLearning.Neural;

/// <summary>TUnit-oriented base that combines <see cref="UnitTestBase"/> output with framework-agnostic neural expectations.</summary>
public abstract class NeuralNetworkBaseTest : UnitTestBase
{
    protected static void ExpectEqual(
        ReadOnlySpan<double> expected,
        ReadOnlySpan<double> actual,
        double tolerance = NeuralNumerics.DefaultTolerance) =>
        NeuralNumerics.ExpectEqual(expected, actual, tolerance);

    protected static void ExpectScalarEqual(
        double expected,
        double actual,
        double tolerance = NeuralNumerics.DefaultTolerance) =>
        NeuralNumerics.ExpectScalarEqual(expected, actual, tolerance);

    /// <summary>Serializes a round-tripped copy of the network for test output (indented JSON snapshot).</summary>
    protected void OutputRoundTrip(DenseNetwork network, string snapshotId = "round-trip")
    {
        var copy = NeuralSnapshotRoundTrip.RoundTrip(network, snapshotId);
        Output(NeuralSnapshotRoundTrip.Serialize(copy, snapshotId));
    }
}
