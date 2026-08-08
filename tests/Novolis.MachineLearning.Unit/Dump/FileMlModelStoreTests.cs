using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

using Novolis.MachineLearning.Dump;

namespace Novolis.MachineLearning.Dump.Tests;

public sealed class FileMlModelStoreTests
{
    private sealed class SampleRow
    {
        public float Feature { get; set; }
        public bool Label { get; set; }
    }

    [Test]
    public async Task Save_And_Load_RoundTripsTransformer()
    {
        var root = Path.Combine(Path.GetTempPath(), "ml-model-store-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            var ml = new MLContext(seed: 1);
            var rows = Enumerable.Range(0, 40).Select(i => new SampleRow
            {
                Feature = i,
                Label = i % 2 == 0,
            });
            var data = ml.Data.LoadFromEnumerable(rows);
            // One SDCA iteration is enough to round-trip a non-null transformer through the store.
            var pipeline = ml.Transforms.Concatenate("Features", nameof(SampleRow.Feature))
                .Append(ml.BinaryClassification.Trainers.SdcaLogisticRegression(
                    new SdcaLogisticRegressionBinaryTrainer.Options
                    {
                        LabelColumnName = nameof(SampleRow.Label),
                        FeatureColumnName = "Features",
                        MaximumNumberOfIterations = 1,
                    }));
            var model = pipeline.Fit(data);

            var store = new FileMlModelStore(root);
            store.Save(ml, model, data.Schema, "binary-sdca");

            await Assert.That(store.ListIds()).Contains("binary-sdca");
            var loaded = store.Load(ml, "binary-sdca");
            await Assert.That(loaded).IsNotNull();
            await Assert.That(store.Delete("binary-sdca")).IsTrue();
        }
        finally
        {
            if (Directory.Exists(root))
                Directory.Delete(root, recursive: true);
        }
    }
}
