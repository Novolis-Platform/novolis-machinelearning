using Novolis.MachineLearning.Llm;

namespace Novolis.MachineLearning.SharpMind.Tests;

public sealed class SharpMindTextTrainerTests
{
    [Test]
    public async Task TrainTextAsync_TinyCorpus_RunsOneTrainingStep()
    {
        var workingDirectory = Path.Combine(Path.GetTempPath(), "novolis-sharpmind-test-" + Guid.NewGuid().ToString("N"));
        try
        {
            var trainer = new SharpMindTextTrainer();
            var result = await trainer.TrainTextAsync(
                [
                    "alpha beta gamma",
                    "alpha beta delta",
                    "beta gamma alpha",
                    "delta alpha beta",
                ],
                new SharpMindTextTrainingOptions
                {
                    WorkingDirectory = workingDirectory,
                    TotalSteps = 1,
                    BatchSize = 1,
                    SequenceLength = 8,
                    LearningRate = 0.001f,
                    VocabularySize = 16,
                    HiddenSize = 16,
                    FeedForwardSize = 32,
                    Layers = 1,
                    Heads = 2,
                    Seed = 7,
                });

            await Assert.That(result.ModelName).IsEqualTo("sharpmind-tiny-text");
            await Assert.That(result.VocabularySize).IsEqualTo(16);
            await Assert.That(result.Steps.Count).IsEqualTo(1);
            await Assert.That(float.IsFinite(result.Steps[0].Loss)).IsTrue();
            await Assert.That(result.Steps[0].Loss).IsGreaterThan(0);
        }
        finally
        {
            if (Directory.Exists(workingDirectory))
                Directory.Delete(workingDirectory, recursive: true);
        }
    }

    [Test]
    public async Task TrainTextAsync_EmptyCorpus_Throws()
    {
        var trainer = new SharpMindTextTrainer();

        async Task<LanguageModelTrainingResult?> Act()
            => await trainer.TrainTextAsync(Array.Empty<string>(), new SharpMindTextTrainingOptions());

        await Assert.That(Act).Throws<ArgumentException>();
    }

    [Test]
    public async Task Validate_HiddenSizeNotDivisibleByHeads_Throws()
    {
        var options = new SharpMindTextTrainingOptions
        {
            HiddenSize = 15,
            Heads = 2,
        };

        var act = options.Validate;

        await Assert.That(act).Throws<ArgumentException>();
    }
}
