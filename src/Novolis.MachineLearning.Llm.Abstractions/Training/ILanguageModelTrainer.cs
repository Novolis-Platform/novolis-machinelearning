namespace Novolis.MachineLearning.Llm;

/// <summary>Provider-neutral language-model trainer contract.</summary>
public interface ILanguageModelTrainer
{
    /// <summary>Runs a small training job from plain text examples.</summary>
    /// <param name="corpus">Plain text training documents or lines.</param>
    /// <param name="options">Training options.</param>
    /// <param name="stepProgress">Optional per-step progress callback.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Training summary.</returns>
    Task<LanguageModelTrainingResult> TrainTextAsync(
        IReadOnlyList<string> corpus,
        LanguageModelTrainingOptions? options = null,
        IProgress<LanguageModelTrainingStep>? stepProgress = null,
        CancellationToken cancellationToken = default);
}
