using Novolis.MachineLearning.Llm;

namespace Novolis.MachineLearning.Llm.Tests;

public sealed class ChatGenerationOptionsTests
{
    [Test]
    public async Task Validate_DefaultOptions_DoesNotThrow()
    {
        ChatGenerationOptions.Default.Validate();
        await Assert.That(ChatGenerationOptions.Default.MaxTokens).IsEqualTo(512);
    }

    [Test]
    public async Task Validate_InvalidTopP_Throws()
    {
        var options = ChatGenerationOptions.Default with { TopP = 1.1f };

        var act = options.Validate;

        await Assert.That(act).Throws<ArgumentOutOfRangeException>();
    }

    [Test]
    public async Task ChatOnlyCapabilities_EnableChatWithoutTraining()
    {
        var capabilities = LanguageModelCapabilities.ChatOnly;

        await Assert.That(capabilities.SupportsChat).IsTrue();
        await Assert.That(capabilities.SupportsTraining).IsFalse();
    }

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
