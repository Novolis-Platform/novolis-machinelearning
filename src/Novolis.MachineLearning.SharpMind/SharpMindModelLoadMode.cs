namespace Novolis.MachineLearning.SharpMind;

/// <summary>SharpMind local model loading strategy.</summary>
public enum SharpMindModelLoadMode
{
    /// <summary>Load all model weights into memory up front.</summary>
    Full,

    /// <summary>Stream model layers from disk during inference.</summary>
    Streaming,
}
