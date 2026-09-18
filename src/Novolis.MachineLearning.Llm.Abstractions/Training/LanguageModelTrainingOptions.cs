namespace Novolis.MachineLearning.Llm;

/// <summary>Provider-neutral options for a small language-model training run.</summary>
public record LanguageModelTrainingOptions
{
    /// <summary>Number of optimization steps to run.</summary>
    public int TotalSteps { get; init; } = 1;

    /// <summary>Sequences per batch.</summary>
    public int BatchSize { get; init; } = 1;

    /// <summary>Maximum training sequence length.</summary>
    public int SequenceLength { get; init; } = 16;

    /// <summary>Learning rate for the optimizer.</summary>
    public float LearningRate { get; init; } = 0.001f;

    /// <summary>Seed for random initialization where supported.</summary>
    public int Seed { get; init; } = 1;

    /// <summary>Directory for provider checkpoints or temporary training artifacts.</summary>
    public string? WorkingDirectory { get; init; }

    /// <summary>Throws when option values are outside supported generic bounds.</summary>
    public void Validate()
    {
        if (TotalSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(TotalSteps), TotalSteps, "TotalSteps must be greater than zero.");
        if (BatchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(BatchSize), BatchSize, "BatchSize must be greater than zero.");
        if (SequenceLength <= 1)
            throw new ArgumentOutOfRangeException(nameof(SequenceLength), SequenceLength, "SequenceLength must be greater than one.");
        if (LearningRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(LearningRate), LearningRate, "LearningRate must be greater than zero.");
    }
}
