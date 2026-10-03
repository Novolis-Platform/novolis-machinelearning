namespace Novolis.MachineLearning.Llm.Tests;

public sealed class TrainingContractsTests
{
    [Test]
    public async Task LanguageModelTrainingResult_ExposesInitialAndFinalLoss()
    {
        var result = new LanguageModelTrainingResult(
            "demo",
            8,
            [
                new LanguageModelTrainingStep(1, 2.5f, 0.1f, 0.3f, TimeSpan.FromMilliseconds(1)),
                new LanguageModelTrainingStep(2, 2.0f, 0.1f, 0.2f, TimeSpan.FromMilliseconds(1)),
            ],
            "work");

        await Assert.That(result.InitialLoss).IsEqualTo(2.5f);
        await Assert.That(result.FinalLoss).IsEqualTo(2.0f);
    }
}
