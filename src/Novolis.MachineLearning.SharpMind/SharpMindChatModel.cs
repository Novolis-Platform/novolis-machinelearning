using System.Text;

using Novolis.MachineLearning.Llm;

using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Inference;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;

using SharpMindChatMessage = SharpMind.Inference.Chat.ChatMessage;
using SharpMindChatMessageRole = SharpMind.Inference.Chat.ChatRole;
using SharpMindChatSession = SharpMind.Inference.Chat.ChatSession<SharpMind.Inference.StandardGeneratorBuilder<SharpMind.Model.KVCacherBuilder>, SharpMind.Model.KVCacherBuilder>;

namespace Novolis.MachineLearning.SharpMind;

/// <summary>SharpMind-backed local chat model.</summary>
public sealed class SharpMindChatModel : IChatModel
{
    private readonly TransformerWeights _weights;
    private readonly SharpMindChatSession _session;
    private bool _disposed;

    private SharpMindChatModel(
        SharpMindChatModelOptions options,
        TransformerWeights weights,
        Transformer model,
        SharpMindChatSession session)
    {
        Options = options;
        _weights = weights;
        _session = session;
        Model = model;
        Name = string.IsNullOrWhiteSpace(options.Name)
            ? Path.GetFileNameWithoutExtension(options.ModelPath)
            : options.Name;
    }

    /// <summary>Options used to load this model.</summary>
    public SharpMindChatModelOptions Options { get; }

    /// <summary>Loaded SharpMind transformer.</summary>
    public Transformer Model { get; }

    /// <inheritdoc />
    public string Name { get; }

    /// <inheritdoc />
    public LanguageModelCapabilities Capabilities { get; } = new()
    {
        SupportsChat = true,
        SupportsLocalModelFiles = true,
        SupportsTokenStreaming = true,
        SupportsTraining = false,
    };

    /// <summary>Loads a local SharpMind-supported model file.</summary>
    /// <param name="options">Load options.</param>
    /// <returns>Loaded chat model.</returns>
    public static SharpMindChatModel Load(SharpMindChatModelOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();

        if (!File.Exists(options.ModelPath))
            throw new FileNotFoundException($"SharpMind model file was not found: {options.ModelPath}", options.ModelPath);

        var extension = Path.GetExtension(options.ModelPath);
        var format = ModelFormatHelpers.GetFormatForExtension(extension)
            ?? throw new NotSupportedException($"SharpMind does not recognize model extension '{extension}'.");

        var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor(format);
        metaHelper.Load(
            options.ModelPath,
            options.TokenizerPath,
            out var meta,
            out var modelConfig,
            out var tokenizer);

        if (tokenizer is null)
            throw new InvalidOperationException("The SharpMind model did not provide tokenizer data and no external tokenizer could be loaded.");

        var hardwareTier = MapHardwareTier(options.HardwareTier);
        var sharpConfig = modelConfig.ForModel(hardwareTier);
        var quantizationOps = QuantizationFactory.Create(sharpConfig.ResolvedHardware);
        var weights = ModelFactory.CreateWeights(
            modelConfig,
            sharpConfig,
            quantizationOps,
            options.ModelPath,
            MapLoadMode(options.LoadMode));

        try
        {
            weights.GgufMeta = meta;
            weights.InitializeWeights();
            var model = ModelFactory.CreateTransformer(weights, sharpConfig);
            var session = new SharpMindChatSession(model, tokenizer, meta, seed: options.Seed);
            ApplyOptions(session, options.GenerationOptions);
            session.InitializeChat();
            return new SharpMindChatModel(options, weights, model, session);
        }
        catch
        {
            weights.Dispose();
            throw;
        }
    }

    /// <inheritdoc />
    public async Task<ChatResponse> GenerateAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatGenerationOptions? options = null,
        IProgress<string>? tokenProgress = null,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(messages);
        if (messages.Count == 0)
            throw new ArgumentException("At least one user message is required.", nameof(messages));

        var generationOptions = options ?? Options.GenerationOptions;
        generationOptions.Validate();

        var lastUserIndex = FindLastUserMessageIndex(messages);
        if (lastUserIndex < 0)
            throw new ArgumentException("The transcript must include at least one user message.", nameof(messages));

        ApplyOptions(_session, generationOptions);
        _session.ClearHistory();
        _session.ResetCaches();

        for (var i = 0; i < lastUserIndex; i++)
            _session.AddMessage(MapRole(messages[i].Role), messages[i].Content);

        var builder = new StringBuilder();
        await foreach (var entry in _session.GetResponseStreamAsync(messages[lastUserIndex].Content, ct: cancellationToken)
                           .ConfigureAwait(false))
        {
            if (entry.Token is not { Length: > 0 } token)
                continue;

            builder.Append(token);
            tokenProgress?.Report(token);
        }

        return new ChatResponse(
            builder.ToString(),
            _session.History.Select(MapMessage).ToArray(),
            _session.TimeToFirstToken,
            _session.TokensPerSecond);
    }

    /// <inheritdoc />
    public async ValueTask DisposeAsync()
    {
        if (_disposed)
            return;

        _disposed = true;
        await _session.DisposeAsync().ConfigureAwait(false);
        _weights.Dispose();
    }

    private static int FindLastUserMessageIndex(IReadOnlyList<ChatMessage> messages)
    {
        for (var i = messages.Count - 1; i >= 0; i--)
        {
            if (messages[i].Role == ChatMessageRole.User)
                return i;
        }

        return -1;
    }

    private static void ApplyOptions(SharpMindChatSession session, ChatGenerationOptions options)
    {
        session.MaxTokens = options.MaxTokens;
        session.Temperature = options.Temperature;
        session.TopK = options.TopK;
        session.TopP = options.TopP;
        session.RepetitionPenalty = options.RepetitionPenalty;
        session.RepetitionWindow = options.RepetitionWindow;
        session.EnableThinking = options.EnableThinking;
        session.ShowThinking = options.ShowThinking;
    }

    private static HardwareTier MapHardwareTier(SharpMindHardwareTier tier) => tier switch
    {
        SharpMindHardwareTier.Auto => HardwareTier.Auto,
        SharpMindHardwareTier.Fma => HardwareTier.FMA,
        SharpMindHardwareTier.Avx2 => HardwareTier.AVX2,
        SharpMindHardwareTier.Sse => HardwareTier.SSE,
        SharpMindHardwareTier.Scalar => HardwareTier.Scalar,
        _ => throw new ArgumentOutOfRangeException(nameof(tier), tier, null),
    };

    private static LoadMode MapLoadMode(SharpMindModelLoadMode loadMode) => loadMode switch
    {
        SharpMindModelLoadMode.Full => LoadMode.Full,
        SharpMindModelLoadMode.Streaming => LoadMode.Streaming,
        _ => throw new ArgumentOutOfRangeException(nameof(loadMode), loadMode, null),
    };

    private static SharpMindChatMessageRole MapRole(ChatMessageRole role) => role switch
    {
        ChatMessageRole.System => SharpMindChatMessageRole.System,
        ChatMessageRole.User => SharpMindChatMessageRole.User,
        ChatMessageRole.Assistant => SharpMindChatMessageRole.Agent,
        _ => throw new ArgumentOutOfRangeException(nameof(role), role, null),
    };

    private static ChatMessage MapMessage(SharpMindChatMessage message)
        => new(MapRole(message.Role), message.Content, message.Name, message.Metadata);

    private static ChatMessageRole MapRole(SharpMindChatMessageRole role) => role switch
    {
        SharpMindChatMessageRole.System => ChatMessageRole.System,
        SharpMindChatMessageRole.User => ChatMessageRole.User,
        SharpMindChatMessageRole.Agent => ChatMessageRole.Assistant,
        _ => throw new ArgumentOutOfRangeException(nameof(role), role, null),
    };

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }
}
