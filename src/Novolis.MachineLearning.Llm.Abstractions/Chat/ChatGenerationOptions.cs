namespace Novolis.MachineLearning.Llm;

/// <summary>Common sampling options for chat generation.</summary>
public sealed record ChatGenerationOptions
{
    /// <summary>Default chat generation options.</summary>
    public static ChatGenerationOptions Default { get; } = new();

    /// <summary>Maximum new tokens to generate.</summary>
    public int MaxTokens { get; init; } = 512;

    /// <summary>Sampling temperature. Use zero for deterministic generation when supported.</summary>
    public float Temperature { get; init; } = 0.7f;

    /// <summary>Top-k sampling limit.</summary>
    public int TopK { get; init; } = 40;

    /// <summary>Top-p nucleus sampling threshold.</summary>
    public float TopP { get; init; } = 0.9f;

    /// <summary>Penalty applied to recently repeated tokens.</summary>
    public float RepetitionPenalty { get; init; } = 1.0f;

    /// <summary>Number of previous tokens considered by repetition penalty.</summary>
    public int RepetitionWindow { get; init; } = 64;

    /// <summary>Whether the provider should enable thinking/reasoning controls when supported.</summary>
    public bool EnableThinking { get; init; }

    /// <summary>Whether thinking/reasoning content should be surfaced when supported.</summary>
    public bool ShowThinking { get; init; }

    /// <summary>Throws when option values are outside supported generic bounds.</summary>
    public void Validate()
    {
        if (MaxTokens <= 0)
            throw new ArgumentOutOfRangeException(nameof(MaxTokens), MaxTokens, "MaxTokens must be greater than zero.");
        if (Temperature < 0)
            throw new ArgumentOutOfRangeException(nameof(Temperature), Temperature, "Temperature cannot be negative.");
        if (TopK < 0)
            throw new ArgumentOutOfRangeException(nameof(TopK), TopK, "TopK cannot be negative.");
        if (TopP is < 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(TopP), TopP, "TopP must be between 0 and 1.");
        if (RepetitionPenalty <= 0)
            throw new ArgumentOutOfRangeException(nameof(RepetitionPenalty), RepetitionPenalty, "RepetitionPenalty must be greater than zero.");
        if (RepetitionWindow < 0)
            throw new ArgumentOutOfRangeException(nameof(RepetitionWindow), RepetitionWindow, "RepetitionWindow cannot be negative.");
    }
}
