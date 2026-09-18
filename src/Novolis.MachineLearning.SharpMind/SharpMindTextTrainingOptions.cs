using Novolis.MachineLearning.Llm;

namespace Novolis.MachineLearning.SharpMind;

/// <summary>SharpMind-specific options for from-scratch text training.</summary>
public sealed record SharpMindTextTrainingOptions : LanguageModelTrainingOptions
{
    /// <summary>Model display name.</summary>
    public string ModelName { get; init; } = "sharpmind-tiny-text";

    /// <summary>Total tokenizer/model vocabulary size, including SharpMind special tokens.</summary>
    public int VocabularySize { get; init; } = 64;

    /// <summary>Hidden transformer dimension for the demo model.</summary>
    public int HiddenSize { get; init; } = 32;

    /// <summary>Feed-forward dimension for the demo model.</summary>
    public int FeedForwardSize { get; init; } = 64;

    /// <summary>Number of transformer layers.</summary>
    public int Layers { get; init; } = 1;

    /// <summary>Number of attention heads.</summary>
    public int Heads { get; init; } = 2;

    /// <summary>Hardware tier override for SharpMind training kernels.</summary>
    public SharpMindHardwareTier HardwareTier { get; init; } = SharpMindHardwareTier.Auto;

    /// <inheritdoc />
    public new void Validate()
    {
        base.Validate();
        ArgumentException.ThrowIfNullOrWhiteSpace(ModelName);
        if (VocabularySize <= 4)
            throw new ArgumentOutOfRangeException(nameof(VocabularySize), VocabularySize, "VocabularySize must leave room for words and special tokens.");
        if (HiddenSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(HiddenSize), HiddenSize, "HiddenSize must be greater than zero.");
        if (FeedForwardSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(FeedForwardSize), FeedForwardSize, "FeedForwardSize must be greater than zero.");
        if (Layers <= 0)
            throw new ArgumentOutOfRangeException(nameof(Layers), Layers, "Layers must be greater than zero.");
        if (Heads <= 0)
            throw new ArgumentOutOfRangeException(nameof(Heads), Heads, "Heads must be greater than zero.");
        if (HiddenSize % Heads != 0)
            throw new ArgumentException("HiddenSize must be divisible by Heads.", nameof(HiddenSize));
    }
}
