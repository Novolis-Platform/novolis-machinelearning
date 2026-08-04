using Microsoft.ML;
using Microsoft.ML.Data;
using System.IO.Abstractions.TestingHelpers;

using Novolis.MachineLearning.AutoMl;
using Novolis.MachineLearning.AutoMl.Extensions;
using Novolis.MachineLearning.Core.Paths;
using Novolis.MachineLearning.Neural.Persistence;

namespace Novolis.MachineLearning.Unit.AutoMl;

public sealed class MetricsExtensionsAndModelSelectorTests
{
    private sealed class HouseRow
    {
        public float Label { get; set; }
        public float Size { get; set; }
        public float Rooms { get; set; }
    }

    private sealed class IrisRow
    {
        public string Label { get; set; } = "";
        public float Feature1 { get; set; }
        public float Feature2 { get; set; }
    }

    private sealed class BinaryRow
    {
        public bool Label { get; set; }
        public float Feature1 { get; set; }
        public float Feature2 { get; set; }
    }

    private sealed class RatingRow
    {
        public float UserId { get; set; }
        public float ItemId { get; set; }
        public float Label { get; set; }
    }

    [Test]
    public async Task ToFriendlyString_Multiclass_IncludesAccuracyAndLogLoss()
    {
        var ml = new MLContext(seed: 0);
        var rows = Enumerable.Range(0, 30).Select(i => new IrisRow
        {
            Label = i % 3 == 0 ? "A" : i % 3 == 1 ? "B" : "C",
            Feature1 = i * 0.1f,
            Feature2 = i * 0.2f,
        }).ToList();
        var data = ml.Data.LoadFromEnumerable(rows);
        var pipeline = ml.Transforms.Conversion.MapValueToKey("Label")
            .Append(ml.Transforms.Concatenate("Features", nameof(IrisRow.Feature1), nameof(IrisRow.Feature2)))
            .Append(ml.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"));
        var model = pipeline.Fit(data);
        var scored = model.Transform(data);
        var metrics = ml.MulticlassClassification.Evaluate(scored, "Label");
        var text = metrics.ToFriendlyString();
        await Assert.That(text).Contains("MacroAccuracy");
        await Assert.That(text).Contains("LogLoss");
    }

    [Test]
    public async Task ToFriendlyString_MulticlassCrossValidation_IncludesAverages()
    {
        var ml = new MLContext(seed: 1);
        var rows = Enumerable.Range(0, 40).Select(i => new IrisRow
        {
            Label = i % 2 == 0 ? "A" : "B",
            Feature1 = i,
            Feature2 = i * 0.5f,
        }).ToList();
        var data = ml.Data.LoadFromEnumerable(rows);
        var pipeline = ml.Transforms.Conversion.MapValueToKey("Label")
            .Append(ml.Transforms.Concatenate("Features", nameof(IrisRow.Feature1), nameof(IrisRow.Feature2)))
            .Append(ml.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"));
        var cv = ml.MulticlassClassification.CrossValidate(data, pipeline, numberOfFolds: 2);
        var text = cv.ToFriendlyString();
        await Assert.That(text).Contains("Average MicroAccuracy");
        await Assert.That(text).Contains("Average LogLossReduction");
    }

    [Test]
    public async Task ToFriendlyString_Regression_AndCrossValidation()
    {
        var ml = new MLContext(seed: 2);
        var rows = Enumerable.Range(1, 25).Select(i => new HouseRow { Label = i * 2f, Size = i, Rooms = i % 4 }).ToList();
        var data = ml.Data.LoadFromEnumerable(rows);
        var pipeline = ml.Transforms.Concatenate("Features", nameof(HouseRow.Size), nameof(HouseRow.Rooms))
            .Append(ml.Regression.Trainers.Sdca(labelColumnName: nameof(HouseRow.Label), featureColumnName: "Features"));
        var model = pipeline.Fit(data);
        var scored = model.Transform(data);
        var metrics = ml.Regression.Evaluate(scored, labelColumnName: nameof(HouseRow.Label));
        var single = metrics.ToFriendlyString();
        await Assert.That(single).Contains("R2 Score");

        var cv = ml.Regression.CrossValidate(data, pipeline, numberOfFolds: 2);
        var cvText = cv.ToFriendlyString();
        await Assert.That(cvText).Contains("Average RMS");
    }

    [Test]
    public async Task ModelSelector_RunsRegressionExperiment()
    {
        var rows = Enumerable.Range(1, 80).Select(i => new HouseRow { Label = i * 1.5f, Size = i, Rooms = i % 3 }).ToList();
        var selector = new ModelSelector<HouseRow>();
        var result = selector.RunRegressionExperiment(rows, nameof(HouseRow.Label), runtime: 15);
        await Assert.That(result.BestRun).IsNotNull();
        await Assert.That(result.BestRun.ValidationMetrics.RSquared).IsGreaterThan(-1);
    }

    [Test]
    public async Task ModelSelector_RunsRegressionWithProgressCallback()
    {
        var rows = Enumerable.Range(1, 80).Select(i => new HouseRow { Label = i * 1.5f, Size = i, Rooms = i % 3 }).ToList();
        var selector = new ModelSelector<HouseRow>();
        var result = selector.RunRegressionExperiment(rows, nameof(HouseRow.Label), runtime: 15, trialProgress: null);
        await Assert.That(result.BestRun).IsNotNull();
    }

    [Test]
    public async Task ModelSelector_RunsBinaryClassificationExperiment()
    {
        var rows = Enumerable.Range(0, 100).Select(i => new BinaryRow
        {
            Label = i % 2 == 0,
            Feature1 = i * 0.1f,
            Feature2 = i * 0.2f,
        }).ToList();
        var selector = new ModelSelector<BinaryRow>();
        var binary = selector.RunBinaryClassificationExperiment(rows, runtime: 20);
        await Assert.That(binary.BestRun).IsNotNull();
    }

    [Test]
    public async Task ModelSelector_RunsMulticlassAndRecommendationExperiments()
    {
        var rows = Enumerable.Range(0, 120).Select(i => new IrisRow
        {
            Label = i % 3 == 0 ? "A" : i % 3 == 1 ? "B" : "C",
            Feature1 = i * 0.05f,
            Feature2 = i * 0.07f,
        }).ToList();
        var ratings = Enumerable.Range(0, 200).Select(i => new RatingRow
        {
            UserId = i % 20,
            ItemId = i % 15,
            Label = (i % 5) + 1f,
        }).ToList();
        var selector = new ModelSelector<IrisRow>();
        var multi = selector.RunMulticlassClassificationExperiment(rows, runtime: 25);
        var recSelector = new ModelSelector<RatingRow>();
        var rec = recSelector.RunRecommendationExperiment(ratings, runtime: 30);
        await Assert.That(multi.BestRun).IsNotNull();
        await Assert.That(rec).IsNotNull();
    }

    [Test]
    public async Task NeuralPreset_StoresMetadata()
    {
        var snapshot = new NetworkSnapshot(
            "test-id",
            "test-net",
            InputSize: 2,
            OutputSize: 1,
            LayerSizes: [2, 1],
            Layers: [],
            CreatedAtUtc: DateTimeOffset.UtcNow);
        var preset = new NeuralPreset("id", "Name", "Family", "Blue", snapshot, new Dictionary<string, string> { ["k"] = "v" });
        await Assert.That(preset.Id).IsEqualTo("id");
        await Assert.That(preset.Metadata!["k"]).IsEqualTo("v");
    }

    [Test]
    public async Task RepoPaths_FallbackRoots_WhenNoRepoMarker()
    {
        var fs = new MockFileSystem(new Dictionary<string, MockFileData>());
        var sessions = NovolisMachineLearningRepoPaths.NeuralLabsRunnerSessionsRoot(fs);
        var settings = NovolisMachineLearningRepoPaths.NeuralLabsSettingsDirectory(fs);
        await Assert.That(sessions).Contains("neural-labs");
        await Assert.That(settings).Contains("settings");
        await Assert.That(NovolisMachineLearningRepoPaths.TryGetRepoRoot(
            Path.GetFullPath(Path.Combine(Path.DirectorySeparatorChar == '\\' ? @"C:\orphan" : "/orphan")),
            fs)).IsNull();
    }

    [Test]
    public async Task RepoPaths_TryGetRelativePath_AndFallbackRoots()
    {
        var tmp = Path.Combine(Path.GetTempPath(), "NovolisMLPaths-" + Guid.NewGuid().ToString("N"));
        var nested = Path.Combine(tmp, "bin");
        try
        {
            Directory.CreateDirectory(nested);
            await File.WriteAllTextAsync(Path.Combine(tmp, "Novolis.MachineLearning.slnx"), "<Solution />");
            var orphan = NovolisMachineLearningRepoPaths.NeuralLabsRunnerOrphanExportsRoot();
            var settings = NovolisMachineLearningRepoPaths.NeuralLabsSettingsDirectory();
            var rel = NovolisMachineLearningRepoPaths.TryGetPathRelativeToRepo(Path.Combine(tmp, "settings", "neural-labs", "x.json"));
            var outside = NovolisMachineLearningRepoPaths.TryGetPathRelativeToRepo(Path.GetTempPath());
            await Assert.That(orphan).Contains("orphan-exports");
            await Assert.That(settings).Contains("neural-labs");
            await Assert.That(rel).Contains("settings");
            await Assert.That(outside).IsEqualTo(Path.GetTempPath());
        }
        finally
        {
            try { Directory.Delete(tmp, true); } catch { /* best effort */ }
        }
    }
}
