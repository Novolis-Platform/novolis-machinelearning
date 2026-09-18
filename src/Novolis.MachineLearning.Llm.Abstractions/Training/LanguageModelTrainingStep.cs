namespace Novolis.MachineLearning.Llm;

/// <summary>One language-model training step measurement.</summary>
/// <param name="Step">One-based training step number reported by the provider.</param>
/// <param name="Loss">Training loss.</param>
/// <param name="LearningRate">Learning rate used for the step.</param>
/// <param name="GradientNorm">Gradient norm reported by the provider.</param>
/// <param name="Duration">Step duration.</param>
public sealed record LanguageModelTrainingStep(
    int Step,
    float Loss,
    float LearningRate,
    float GradientNorm,
    TimeSpan Duration);
