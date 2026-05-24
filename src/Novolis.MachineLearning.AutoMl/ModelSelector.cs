using System.Collections.Generic;

using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

namespace Novolis.MachineLearning.AutoMl;

/// <summary>ML.NET AutoML experiment runner for tabular data.</summary>
/// <typeparam name="TEntity">Training row type.</typeparam>
public class ModelSelector<TEntity> where TEntity : class, new()
{
    private readonly MLContext _mlContext;

    /// <summary>Creates a selector with a default ML context.</summary>
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

    /// <summary>Runs a multiclass classification AutoML experiment.</summary>
    public ExperimentResult<MulticlassClassificationMetrics> RunMulticlassClassificationExperiment(IEnumerable<TEntity> items, string labelColumn = "Label", uint runtime = 10)
    {
        var data = _mlContext.Data.LoadFromEnumerable(items);

        var experiment = _mlContext.Auto().CreateMulticlassClassificationExperiment(runtime);
        var experimentResult = experiment.Execute(data, labelColumn);
        return experimentResult;
    }

    /// <summary>Runs a binary classification AutoML experiment.</summary>
    public ExperimentResult<BinaryClassificationMetrics> RunBinaryClassificationExperiment(IEnumerable<TEntity> items, string labelColumn = "Label", uint runtime = 10)
    {
        var data = _mlContext.Data.LoadFromEnumerable(items);

        var experiment = _mlContext.Auto().CreateBinaryClassificationExperiment(runtime);
        var experimentResult = experiment.Execute(data, labelColumn);
        return experimentResult;
    }

    /// <summary>Runs a recommendation AutoML experiment.</summary>
    public ExperimentResult<RegressionMetrics> RunRecommendationExperiment(IEnumerable<TEntity> items, string labelColumn = "Label", uint runtime = 10)
    {
        var data = _mlContext.Data.LoadFromEnumerable(items);

        var experiment = _mlContext.Auto().CreateRecommendationExperiment(runtime);
        var experimentResult = experiment.Execute(data, labelColumn);
        return experimentResult;
    }

    /// <summary>Runs a regression experiment without per-trial progress.</summary>
    public ExperimentResult<RegressionMetrics> RunRegressionExperiment(
        IEnumerable<TEntity> items,
        string labelColumn = "Label",
        uint runtime = 10)
        => RunRegressionExperiment(items, labelColumn, runtime, trialProgress: null);
}
