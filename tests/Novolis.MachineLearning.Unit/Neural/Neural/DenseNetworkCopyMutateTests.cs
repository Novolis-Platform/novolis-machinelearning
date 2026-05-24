namespace Novolis.MachineLearning.Neural;

using Novolis.MachineLearning.Neural;

using TUnit.Assertions;

public sealed class DenseNetworkCopyMutateTests
{
    [Test]
    public async Task Clone_IsIndependent_MutatingCloneDoesNotChangeOriginal()
    {
        var original = DenseNetwork.Create("o", 3, [4], 2, random: new Random(99));
        var before = original.Forward(new double[] { 0.1, 0.2, 0.3 }).ToArray();
        var clone = (DenseNetwork)original.Clone("c");
        clone.Mutate(new Random(1), new MutationSettings(1, 0.5, 1, 0.5));
        var afterOriginal = original.Forward(new double[] { 0.1, 0.2, 0.3 }).ToArray();
        await Assert.That(afterOriginal).IsEquivalentTo(before);
    }

    [Test]
    public async Task Clone_WithName_OverridesName()
    {
        var original = DenseNetwork.Create("old", 2, [], 1, random: new Random(0));
        var clone = (DenseNetwork)original.Clone("new-name");
        await Assert.That(clone.Name).IsEqualTo("new-name");
    }

    [Test]
    public async Task CopyFrom_SameTopology_CopiesWeightsAndBiases()
    {
        var a = DenseNetwork.Create("a", 2, [3], 1, random: new Random(5));
        var b = DenseNetwork.Create("b", 2, [3], 1, random: new Random(77));
        b.CopyFrom(a);
        var input = new double[] { -1, 2 };
        await Assert.That(b.Forward(input).ToArray()).IsEquivalentTo(a.Forward(input).ToArray());
    }

    [Test]
    public async Task CopyFrom_InputSizeMismatch_Throws()
    {
        var a = DenseNetwork.Create("a", 2, [], 1);
        var b = DenseNetwork.Create("b", 3, [], 1);
        await Assert.That(() => b.CopyFrom(a)).Throws<InvalidOperationException>()
            .WithMessage("Network shapes do not match.");
    }

    [Test]
    public async Task CopyFrom_OutputSizeMismatch_Throws()
    {
        var a = DenseNetwork.Create("a", 2, [], 1);
        var b = DenseNetwork.Create("b", 2, [], 2);
        await Assert.That(() => b.CopyFrom(a)).Throws<InvalidOperationException>();
    }

    [Test]
    public async Task CopyFrom_HiddenTopologyMismatch_Throws()
    {
        var a = DenseNetwork.Create("a", 2, [4], 1);
        var b = DenseNetwork.Create("b", 2, [5], 1);
        await Assert.That(() => b.CopyFrom(a)).Throws<InvalidOperationException>();
    }

    [Test]
    public async Task Mutate_WithCertainRates_ChangesAtLeastOneParameter()
    {
        var net = DenseNetwork.Create("m", 4, [5, 5], 2, random: new Random(0));
        var flatBefore = Serialize(net);
        net.Mutate(new Random(123), new MutationSettings(
            WeightMutationRate: 1,
            WeightMutationSigma: 0.01,
            BiasMutationRate: 1,
            BiasMutationSigma: 0.01));
        await Assert.That(Serialize(net)).IsNotEqualTo(flatBefore);
    }

    private static string Serialize(DenseNetwork net)
    {
        var parts = new List<string>();
        foreach (var layer in net.Layers)
        {
            for (var i = 0; i < layer.InputCount; i++)
                for (var j = 0; j < layer.OutputCount; j++)
                    parts.Add(layer.Weights[i, j].ToString("R"));
            foreach (var b in layer.Biases)
                parts.Add(b.ToString("R"));
        }

        return string.Join("|", parts);
    }
}
