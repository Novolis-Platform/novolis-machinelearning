using Novolis.MachineLearning.AutoMl.Extensions;

namespace Novolis.MachineLearning.Unit.AutoMl;

public sealed class MetricsExtensionsTests
{
    [Test]
    public async Task CalculateStandardDeviation_is_zero_for_identical_values()
    {
        var values = new[] { 3d, 3, 3, 3 };
        var deviation = MetricsExtensions.CalculateStandardDeviation(values);
        await Assert.That(deviation).IsEqualTo(0);
    }

    [Test]
    public async Task CalculateConfidenceInterval95_is_positive_for_multiple_values()
    {
        var values = new[] { 0.8, 0.85, 0.9, 0.75 };
        var interval = MetricsExtensions.CalculateStandardDeviation(values);
        var confidence = MetricsExtensions.CalculateConfidenceInterval95(values);

        await Assert.That(interval).IsGreaterThan(0);
        await Assert.That(confidence).IsGreaterThan(0);
    }
}
