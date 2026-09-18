namespace Novolis.MachineLearning.Llm;

/// <summary>Provider-neutral chat message.</summary>
/// <param name="Role">Message role.</param>
/// <param name="Content">Message content.</param>
/// <param name="Name">Optional participant name.</param>
/// <param name="Metadata">Optional provider-neutral metadata.</param>
public sealed record ChatMessage(
    ChatMessageRole Role,
    string Content,
    string? Name = null,
    IReadOnlyDictionary<string, string>? Metadata = null);
