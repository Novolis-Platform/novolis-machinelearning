using Novolis.MachineLearning.Llm;

namespace Novolis.MachineLearning.SharpMind.Tests;

public sealed class SharpMindChatModelOptionsTests
{
    [Test]
    public async Task Validate_MissingModelPath_Throws()
    {
        var options = new SharpMindChatModelOptions { ModelPath = "" };

        var act = options.Validate;

        await Assert.That(act).Throws<ArgumentException>();
    }

    [Test]
    public async Task Load_MissingModelFile_ThrowsBeforeSharpMindLoading()
    {
        var missing = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".gguf");
        var options = new SharpMindChatModelOptions { ModelPath = missing };

        var act = () => SharpMindChatModel.Load(options);

        await Assert.That(act).Throws<FileNotFoundException>();
    }

    [Test]
    public async Task Validate_DefaultGenerationOptions_DoesNotThrow()
    {
        var options = new SharpMindChatModelOptions
        {
            ModelPath = "placeholder.gguf",
            GenerationOptions = ChatGenerationOptions.Default
        };

        options.Validate();
        await Assert.That(options.GenerationOptions.MaxTokens).IsEqualTo(512);
    }
}
