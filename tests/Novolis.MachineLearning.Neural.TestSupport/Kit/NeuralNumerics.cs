using System.Globalization;

namespace Novolis.MachineLearning.Neural.Kit;

/// <summary>Double vector comparisons with no dependency on a test framework.</summary>
public static class NeuralNumerics
{
    public const double DefaultTolerance = 1e-9;

    public static double MaxAbsDiff(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        if (a.Length != b.Length)
            throw new NeuralAssertionException($"Length mismatch: {a.Length} vs {b.Length}.");

        var max = 0.0;
        for (var i = 0; i < a.Length; i++)
        {
            var d = Math.Abs(a[i] - b[i]);
            if (d > max)
                max = d;
        }

        return max;
    }

    public static void ExpectEqual(
        ReadOnlySpan<double> expected,
        ReadOnlySpan<double> actual,
        double tolerance = DefaultTolerance)
    {
        if (tolerance < 0)
            throw new ArgumentOutOfRangeException(nameof(tolerance));

        var diff = MaxAbsDiff(expected, actual);
        if (double.IsNaN(diff) || diff > tolerance)
        {
            static string F(ReadOnlySpan<double> v) =>
                string.Join(", ", v.ToArray().Select(x => x.ToString("G17", CultureInfo.InvariantCulture)));

            throw new NeuralAssertionException(
                $"Vector mismatch (max abs diff {diff:G17}, tolerance {tolerance:G17}). Expected [{F(expected)}] Actual [{F(actual)}]");
        }
    }

    public static void ExpectScalarEqual(double expected, double actual, double tolerance = DefaultTolerance)
    {
        var diff = Math.Abs(expected - actual);
        if (double.IsNaN(diff) || diff > tolerance)
            throw new NeuralAssertionException(
                $"Scalar mismatch: expected {expected:G17}, actual {actual:G17} (diff {diff:G17}, tolerance {tolerance:G17}).");
    }
}
