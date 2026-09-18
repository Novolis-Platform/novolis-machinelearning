namespace Novolis.MachineLearning.Llm;

/// <summary>Provider-neutral chat model contract.</summary>
public interface IChatModel : IAsyncDisposable
{
    /// <summary>Display name for diagnostics and UI.</summary>
    string Name { get; }

    /// <summary>Capabilities exposed by this model backend.</summary>
    LanguageModelCapabilities Capabilities { get; }

    /// <summary>Generates one assistant response for the supplied transcript.</summary>
    /// <param name="messages">Transcript ending in a user message.</param>
    /// <param name="options">Optional generation settings.</param>
    /// <param name="tokenProgress">Optional generated-token callback.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Generated response and updated transcript.</returns>
    Task<ChatResponse> GenerateAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatGenerationOptions? options = null,
        IProgress<string>? tokenProgress = null,
        CancellationToken cancellationToken = default);
}
