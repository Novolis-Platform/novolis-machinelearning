using Novolis.MachineLearning.Algorithms;

namespace Novolis.MachineLearning.Algorithms.Tests;

public sealed class ClassicTrainersTests
{
    private sealed class MulticlassRow
    {
        public float F1 { get; set; }
        public float F2 { get; set; }
        public string Label { get; set; } = "";
    }

    private sealed class BinaryRow
    {
        public float F1 { get; set; }
        public float F2 { get; set; }
        public bool Label { get; set; }
    }

    private sealed class RegressionRow
    {
        public float F1 { get; set; }
        public float F2 { get; set; }
        public float Label { get; set; }
    }

    [Test]
    public async Task FitNaiveBayes_ProducesTransformer()
    {
        // Naive Bayes expects binary (0/1) features.
        var rows = new List<MulticlassRow>
        {
            new() { F1 = 1, F2 = 0, Label = "a" },
            new() { F1 = 1, F2 = 1, Label = "a" },
            new() { F1 = 0, F2 = 1, Label = "b" },
            new() { F1 = 0, F2 = 0, Label = "b" },
            new() { F1 = 1, F2 = 0, Label = "a" },
            new() { F1 = 0, F2 = 1, Label = "b" },
        };

        var trainers = new ClassicTrainers();
        var model = trainers.FitNaiveBayes(rows, nameof(MulticlassRow.Label), nameof(MulticlassRow.F1), nameof(MulticlassRow.F2));
        await Assert.That(model).IsNotNull();
    }

    [Test]
    public async Task FitSdcaBinary_And_FastTreeRegression_ProduceTransformers()
    {
        var binary = Enumerable.Range(0, 50).Select(i => new BinaryRow
        {
            F1 = i,
            F2 = i * 0.5f,
            Label = i % 2 == 0,
        });
        var regression = Enumerable.Range(0, 50).Select(i => new RegressionRow
        {
            F1 = i,
            F2 = i * 0.25f,
            Label = i * 1.5f,
        });

        var trainers = new ClassicTrainers();
        var binaryModel = trainers.FitSdcaBinary(binary, nameof(BinaryRow.Label), nameof(BinaryRow.F1), nameof(BinaryRow.F2));
        var treeModel = trainers.FitFastTreeRegression(regression, nameof(RegressionRow.Label), nameof(RegressionRow.F1), nameof(RegressionRow.F2));

        await Assert.That(binaryModel).IsNotNull();
        await Assert.That(treeModel).IsNotNull();
    }

    [Test]
    public async Task FitLightGbmBinary_ProducesTransformer()
    {
        var rows = Enumerable.Range(0, 60).Select(i => new BinaryRow
        {
            F1 = i,
            F2 = (i * 3) % 7,
            Label = i % 3 != 0,
        });

        var trainers = new ClassicTrainers();
        var model = trainers.FitLightGbmBinary(rows, nameof(BinaryRow.Label), nameof(BinaryRow.F1), nameof(BinaryRow.F2));
        await Assert.That(model).IsNotNull();
    }
}
