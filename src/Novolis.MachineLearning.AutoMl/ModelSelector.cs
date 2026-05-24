using System.Collections.Generic;

using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

namespace Novolis.MachineLearning.AutoMl;

public class ModelSelector<TEntity> where TEntity : class, new()
{
    private readonly MLContext _mlContext;

    public ModelSelector()
    {
        _mlContext = new MLContext();
    }

    /// <summary>Regression AutoML with optional per-trial progress (e.g. Spectre live UI).</summary>
    public ExperimentResult<RegressionMetrics> RunRegressionExperiment(
        IEnumerable<TEntity> items,
        string labelColumn,
        uint runtime,
        IProgress<RunDetail<RegressionMetrics>>? trialProgress)
    {
        var data = _mlContext.Data.LoadFromEnumerable(items);
        var experiment = _mlContext.Auto().CreateRegressionExperiment(runtime);
        return experiment.Execute(data, labelColumn, null, null, trialProgress);
    }

    public ExperimentResult<MulticlassClassificationMetrics> RunMulticlassClassificationExperiment(IEnumerable<TEntity> items, string labelColumn = "Label", uint runtime = 10)
    {
        var data = _mlContext.Data.LoadFromEnumerable(items);

        var experiment = _mlContext.Auto().CreateMulticlassClassificationExperiment(runtime);
        var experimentResult = experiment.Execute(data, labelColumn);
        return experimentResult;
    }

    public ExperimentResult<BinaryClassificationMetrics> RunBinaryClassificationExperiment(IEnumerable<TEntity> items, string labelColumn = "Label", uint runtime = 10)
    {
        var data = _mlContext.Data.LoadFromEnumerable(items);

        var experiment = _mlContext.Auto().CreateBinaryClassificationExperiment(runtime);
        var experimentResult = experiment.Execute(data, labelColumn);
        return experimentResult;
    }

    public ExperimentResult<RegressionMetrics> RunRecommendationExperiment(IEnumerable<TEntity> items, string labelColumn = "Label", uint runtime = 10)
    {
        var data = _mlContext.Data.LoadFromEnumerable(items);

        var experiment = _mlContext.Auto().CreateRecommendationExperiment(runtime);
        var experimentResult = experiment.Execute(data, labelColumn);
        return experimentResult;
    }

    public ExperimentResult<RegressionMetrics> RunRegressionExperiment(
        IEnumerable<TEntity> items,
        string labelColumn = "Label",
        uint runtime = 10)
        => RunRegressionExperiment(items, labelColumn, runtime, trialProgress: null);
}
