using Novolis.MachineLearning.Neural.Kit;

namespace Novolis.MachineLearning.Unit.Neural.Kit;

public sealed class NeuralNumericsTests
{
    [Test]
    public async Task MaxAbsDiff_ReturnsLargestElementDelta()
    {
        var diff = NeuralNumerics.MaxAbsDiff([1.0, 2.0, 3.0], [1.0, 2.5, 3.0]);
        await Assert.That(diff).IsEqualTo(0.5);
    }

    [Test]
    public async Task MaxAbsDiff_LengthMismatch_Throws()
    {
        var act = () => NeuralNumerics.MaxAbsDiff([1.0], [1.0, 2.0]);
        await Assert.That(act).Throws<NeuralAssertionException>();
    }

    [Test]
    public void ExpectEqual_PassesWithinTolerance()
    {
        NeuralNumerics.ExpectEqual([1.0, 2.0], [1.0 + 1e-12, 2.0 - 1e-12]);
    }

    [Test]
    public async Task ExpectEqual_Mismatch_ThrowsWithMessage()
    {
        var act = () => NeuralNumerics.ExpectEqual([1.0, 0.0], [2.0, 0.0], tolerance: 1e-9);
        await Assert.That(act).Throws<NeuralAssertionException>();
    }

    [Test]
    public async Task ExpectEqual_NegativeTolerance_Throws()
    {
        var act = () => NeuralNumerics.ExpectEqual([1.0], [1.0], tolerance: -1);
        await Assert.That(act).Throws<ArgumentOutOfRangeException>();
    }

    [Test]
    public async Task ExpectScalarEqual_PassesAndFails()
    {
        NeuralNumerics.ExpectScalarEqual(1.0, 1.0 + 1e-12);
        var act = () => NeuralNumerics.ExpectScalarEqual(1.0, 1.5, tolerance: 1e-9);
        await Assert.That(act).Throws<NeuralAssertionException>();
    }

    [Test]
    public async Task NeuralAssertionException_CarriesMessage()
    {
        var ex = new NeuralAssertionException("boom");
        await Assert.That(ex.Message).IsEqualTo("boom");
    }
}
