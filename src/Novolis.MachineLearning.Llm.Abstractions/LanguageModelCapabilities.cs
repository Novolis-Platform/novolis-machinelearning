namespace Novolis.MachineLearning.Llm;

/// <summary>High-level capabilities for a language model backend.</summary>
public sealed record LanguageModelCapabilities
{
    /// <summary>Default chat-only capabilities.</summary>
    public static LanguageModelCapabilities ChatOnly { get; } = new() { SupportsChat = true };

    /// <summary>Whether chat generation is supported.</summary>
    public bool SupportsChat { get; init; }

    /// <summary>Whether token streaming callbacks are supported.</summary>
    public bool SupportsTokenStreaming { get; init; } = true;

    /// <summary>Whether local model files can be loaded by this backend.</summary>
    public bool SupportsLocalModelFiles { get; init; }

    /// <summary>Whether the backend exposes LLM training or fine-tuning operations.</summary>
    public bool SupportsTraining { get; init; }
}
