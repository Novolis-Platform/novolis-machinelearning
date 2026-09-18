using System.Text.RegularExpressions;

using Novolis.MachineLearning.Llm;

using SharpMind.Core;
using SharpMind.Data;
using SharpMind.Data.Batching;
using SharpMind.Data.Pipeline;
using SharpMind.Data.Sources;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Training;
using SharpMind.Training.Loss;
using SharpMind.Training.Optimizers;
using SharpMind.Training.Schedulers;

namespace Novolis.MachineLearning.SharpMind;

/// <summary>Small from-scratch SharpMind text trainer.</summary>
public sealed class SharpMindTextTrainer : ILanguageModelTrainer
{
    private static readonly Regex WordPattern = new(@"[A-Za-z0-9_'-]+", RegexOptions.Compiled);

    /// <inheritdoc />
    public Task<LanguageModelTrainingResult> TrainTextAsync(
        IReadOnlyList<string> corpus,
        LanguageModelTrainingOptions? options = null,
        IProgress<LanguageModelTrainingStep>? stepProgress = null,
        CancellationToken cancellationToken = default)
    {
        var sharpOptions = options switch
        {
            null => new SharpMindTextTrainingOptions(),
            SharpMindTextTrainingOptions typed => typed,
            _ => new SharpMindTextTrainingOptions
            {
                TotalSteps = options.TotalSteps,
                BatchSize = options.BatchSize,
                SequenceLength = options.SequenceLength,
                LearningRate = options.LearningRate,
                Seed = options.Seed,
                WorkingDirectory = options.WorkingDirectory,
            },
        };

        return TrainTextAsync(corpus, sharpOptions, stepProgress, cancellationToken);
    }

    /// <summary>Runs a small from-scratch SharpMind training job from plain text.</summary>
    public async Task<LanguageModelTrainingResult> TrainTextAsync(
        IReadOnlyList<string> corpus,
        SharpMindTextTrainingOptions options,
        IProgress<LanguageModelTrainingStep>? stepProgress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(corpus);
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();

        if (corpus.Count == 0 || corpus.All(string.IsNullOrWhiteSpace))
            throw new ArgumentException("At least one non-empty training line is required.", nameof(corpus));

        var workingDirectory = options.WorkingDirectory
            ?? Path.Combine(Path.GetTempPath(), "novolis-sharpmind-train-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(workingDirectory);

        var corpusPath = Path.Combine(workingDirectory, "corpus.txt");
        await File.WriteAllLinesAsync(corpusPath, corpus, cancellationToken).ConfigureAwait(false);

        var words = ExtractVocabulary(corpus, options.VocabularySize);
        var tokenizer = TrainingTokenizerBuilder.BuildForVocab(words, options.VocabularySize, i => $"<extra_{i}>");

        using var weights = ModelFactory.CreateForTraining(CreateModelConfig(options), CreateSharpMindConfig(options));
        WeightInitializer.InitializeRandomly(weights, options.Seed);

        using var model = ModelFactory.CreateTrainingTransformer(weights, CreateSharpMindConfig(options));
        var parameters = model.Parameters().ToArray();
        var ops = TrainingOpsFactory.Create(CreateSharpMindConfig(options));
        using var optimizer = new AdamW(parameters, ops, options.LearningRate, weightDecay: 0f);
        var scheduler = new ConstantScheduler(options.LearningRate);
        var loader = CreateDataLoader(corpusPath, text => TokenizeWords(text, tokenizer), tokenizer.EosId, tokenizer.PadId, options);
        var trainLoop = new TrainLoop(
            model,
            parameters,
            loader,
            optimizer,
            scheduler,
            ops,
            new CrossEntropyLoss(),
            config: new TrainConfig
            {
                TotalSteps = options.TotalSteps,
                GradAccumSteps = 1,
                GradClipNorm = 1.0f,
                LogInterval = 1,
                CheckpointDir = Path.Combine(workingDirectory, "checkpoints"),
                CheckpointInterval = int.MaxValue,
                KeepRecent = 1,
            });

        var steps = new List<LanguageModelTrainingStep>(options.TotalSteps);
        await trainLoop.RunAsync(
            onStep: step =>
            {
                var mapped = new LanguageModelTrainingStep(
                    step.Step,
                    step.Loss,
                    step.LearningRate,
                    step.GradNorm,
                    step.StepTime);
                steps.Add(mapped);
                stepProgress?.Report(mapped);
            },
            cancellationToken: cancellationToken).ConfigureAwait(false);

        return new LanguageModelTrainingResult(
            options.ModelName,
            tokenizer.VocabSize,
            steps,
            workingDirectory);
    }

    private static ModelConfig CreateModelConfig(SharpMindTextTrainingOptions options)
    {
        var config = ModelConfig.Tiny with
        {
            VocabSize = options.VocabularySize,
            MaxSeqLen = options.SequenceLength,
        };
        config.Validate();
        return config;
    }

    private static SharpMindConfig CreateSharpMindConfig(SharpMindTextTrainingOptions options)
        => CreateModelConfig(options).ForModel(MapHardwareTier(options.HardwareTier));

    private static DataLoader CreateDataLoader(
        string corpusPath,
        Func<string, int[]> tokenise,
        int eosTokenId,
        int padTokenId,
        SharpMindTextTrainingOptions options)
    {
        var source = new TextFileSource(corpusPath);
        var pipeline = PipelineNode.From(source);
        var batcher = new PackingBatcher(options.BatchSize, options.SequenceLength, eosTokenId, padTokenId);
        return new DataLoader(pipeline, tokenise, batcher, maxBatches: options.TotalSteps);
    }

    private static IReadOnlyList<string> ExtractVocabulary(IReadOnlyList<string> corpus, int vocabularySize)
    {
        var maxWords = vocabularySize - 4;
        var words = corpus
            .SelectMany(line => WordPattern.Matches(line).Select(match => match.Value))
            .Where(word => !string.IsNullOrWhiteSpace(word))
            .Select(word => word.ToLowerInvariant())
            .Distinct(StringComparer.Ordinal)
            .Take(maxWords)
            .ToArray();

        if (words.Length == 0)
            throw new ArgumentException("The corpus did not contain any tokenizer words.", nameof(corpus));

        return words;
    }

    private static int[] TokenizeWords(string text, global::SharpMind.Tokenization.Tokenizer tokenizer)
        => WordPattern.Matches(text)
            .Select(match =>
            {
                var id = tokenizer.TokenToId(match.Value.ToLowerInvariant());
                return id >= 0 ? id : tokenizer.UnkId;
            })
            .ToArray();

    private static HardwareTier MapHardwareTier(SharpMindHardwareTier tier) => tier switch
    {
        SharpMindHardwareTier.Auto => HardwareTier.Auto,
        SharpMindHardwareTier.Fma => HardwareTier.FMA,
        SharpMindHardwareTier.Avx2 => HardwareTier.AVX2,
        SharpMindHardwareTier.Sse => HardwareTier.SSE,
        SharpMindHardwareTier.Scalar => HardwareTier.Scalar,
        _ => throw new ArgumentOutOfRangeException(nameof(tier), tier, null),
    };
}
