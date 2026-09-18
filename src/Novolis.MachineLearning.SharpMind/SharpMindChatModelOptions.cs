using Novolis.MachineLearning.Llm;

namespace Novolis.MachineLearning.SharpMind;

/// <summary>Options for loading a SharpMind-backed local chat model.</summary>
public sealed record SharpMindChatModelOptions
{
    /// <summary>Path to a SharpMind-supported local model file, such as GGUF or SMM.</summary>
    public required string ModelPath { get; init; }

    /// <summary>Optional external tokenizer path when the model file does not contain tokenizer data.</summary>
    public string? TokenizerPath { get; init; }

    /// <summary>Model loading strategy.</summary>
    public SharpMindModelLoadMode LoadMode { get; init; } = SharpMindModelLoadMode.Full;

    /// <summary>Hardware tier override.</summary>
    public SharpMindHardwareTier HardwareTier { get; init; } = SharpMindHardwareTier.Auto;

    /// <summary>Optional display name. Defaults to the model file name without extension.</summary>
    public string? Name { get; init; }

    /// <summary>Optional generator seed.</summary>
    public int? Seed { get; init; }

    /// <summary>Default generation options applied when a request does not provide overrides.</summary>
    public ChatGenerationOptions GenerationOptions { get; init; } = ChatGenerationOptions.Default;

    /// <summary>Throws when required options are missing or invalid.</summary>
    public void Validate()
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(ModelPath);
        GenerationOptions.Validate();
    }
}
