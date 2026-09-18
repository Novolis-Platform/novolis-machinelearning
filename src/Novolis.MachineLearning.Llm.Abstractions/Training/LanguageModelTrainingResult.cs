namespace Novolis.MachineLearning.Llm;

/// <summary>Summary of a language-model training run.</summary>
/// <param name="ModelName">Trained model name.</param>
/// <param name="VocabularySize">Vocabulary size used by the run.</param>
/// <param name="Steps">Step measurements.</param>
/// <param name="WorkingDirectory">Directory used for temporary artifacts or checkpoints.</param>
public sealed record LanguageModelTrainingResult(
    string ModelName,
    int VocabularySize,
    IReadOnlyList<LanguageModelTrainingStep> Steps,
    string WorkingDirectory)
{
    /// <summary>First reported loss, when available.</summary>
    public float? InitialLoss => Steps.Count == 0 ? null : Steps[0].Loss;

    /// <summary>Last reported loss, when available.</summary>
    public float? FinalLoss => Steps.Count == 0 ? null : Steps[^1].Loss;
}
