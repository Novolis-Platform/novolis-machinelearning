namespace Novolis.MachineLearning.Llm;

/// <summary>Result from a chat generation request.</summary>
/// <param name="Content">Generated assistant text.</param>
/// <param name="History">Provider-neutral transcript after generation, when available.</param>
/// <param name="TimeToFirstToken">Seconds to first generated token, when reported.</param>
/// <param name="TokensPerSecond">Generation throughput, when reported.</param>
public sealed record ChatResponse(
    string Content,
    IReadOnlyList<ChatMessage> History,
    float? TimeToFirstToken = null,
    float? TokensPerSecond = null);
