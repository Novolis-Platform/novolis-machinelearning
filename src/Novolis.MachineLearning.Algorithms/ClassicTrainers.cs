using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;

namespace Novolis.MachineLearning.Algorithms;

/// <summary>
/// Convenience wrappers for common ML.NET classic trainers (Naive Bayes, trees, SDCA, LightGBM).
/// Trainer options default to short budgets suitable for smoke/integration use; tune via ML.NET directly for production training.
/// </summary>
public sealed class ClassicTrainers
{
    private readonly MLContext _mlContext;

    /// <summary>Creates trainers with a default <see cref="MLContext"/>.</summary>
    public ClassicTrainers()
        : this(new MLContext(seed: 1))
    {
    }

    /// <summary>Creates trainers with an existing context.</summary>
    /// <param name="mlContext">Shared ML.NET context.</param>
    public ClassicTrainers(MLContext mlContext)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        _mlContext = mlContext;
    }

    /// <summary>Underlying ML.NET context.</summary>
    public MLContext Context => _mlContext;

    /// <summary>Fits a multiclass Naive Bayes pipeline (features concatenated + trainer).</summary>
    /// <typeparam name="TRow">Input row type.</typeparam>
    /// <param name="rows">Training rows.</param>
    /// <param name="labelColumn">Label column name.</param>
    /// <param name="featureColumns">Feature column names to concatenate.</param>
    /// <returns>Fitted transformer.</returns>
    public ITransformer FitNaiveBayes<TRow>(
        IEnumerable<TRow> rows,
        string labelColumn,
        params string[] featureColumns)
        where TRow : class
    {
        var data = Load(rows);
        var pipeline = FeaturePipeline(featureColumns)
            .Append(_mlContext.Transforms.Conversion.MapValueToKey(labelColumn))
            .Append(_mlContext.MulticlassClassification.Trainers.NaiveBayes(
                labelColumnName: labelColumn,
                featureColumnName: "Features"));
        return pipeline.Fit(data);
    }

    /// <summary>Fits SDCA multiclass logistic regression.</summary>
    /// <typeparam name="TRow">Input row type.</typeparam>
    /// <param name="rows">Training rows.</param>
    /// <param name="labelColumn">Label column name.</param>
    /// <param name="featureColumns">Feature column names to concatenate.</param>
    /// <returns>Fitted transformer.</returns>
    public ITransformer FitSdcaMulticlass<TRow>(
        IEnumerable<TRow> rows,
        string labelColumn,
        params string[] featureColumns)
        where TRow : class
    {
        var data = Load(rows);
        var pipeline = FeaturePipeline(featureColumns)
            .Append(_mlContext.Transforms.Conversion.MapValueToKey(labelColumn))
            .Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                new SdcaMaximumEntropyMulticlassTrainer.Options
                {
                    LabelColumnName = labelColumn,
                    FeatureColumnName = "Features",
                    MaximumNumberOfIterations = 1,
                }));
        return pipeline.Fit(data);
    }

    /// <summary>Fits FastTree binary classification.</summary>
    /// <typeparam name="TRow">Input row type.</typeparam>
    /// <param name="rows">Training rows.</param>
    /// <param name="labelColumn">Label column name.</param>
    /// <param name="featureColumns">Feature column names to concatenate.</param>
    /// <returns>Fitted transformer.</returns>
    public ITransformer FitFastTreeBinary<TRow>(
        IEnumerable<TRow> rows,
        string labelColumn,
        params string[] featureColumns)
        where TRow : class
    {
        var data = Load(rows);
        var pipeline = FeaturePipeline(featureColumns)
            .Append(_mlContext.BinaryClassification.Trainers.FastTree(new FastTreeBinaryTrainer.Options
            {
                LabelColumnName = labelColumn,
                FeatureColumnName = "Features",
                NumberOfTrees = 2,
                NumberOfLeaves = 4,
                MinimumExampleCountPerLeaf = 1,
            }));
        return pipeline.Fit(data);
    }

    /// <summary>Fits FastTree regression.</summary>
    /// <typeparam name="TRow">Input row type.</typeparam>
    /// <param name="rows">Training rows.</param>
    /// <param name="labelColumn">Label column name.</param>
    /// <param name="featureColumns">Feature column names to concatenate.</param>
    /// <returns>Fitted transformer.</returns>
    public ITransformer FitFastTreeRegression<TRow>(
        IEnumerable<TRow> rows,
        string labelColumn,
        params string[] featureColumns)
        where TRow : class
    {
        var data = Load(rows);
        var pipeline = FeaturePipeline(featureColumns)
            .Append(_mlContext.Regression.Trainers.FastTree(new FastTreeRegressionTrainer.Options
            {
                LabelColumnName = labelColumn,
                FeatureColumnName = "Features",
                NumberOfTrees = 2,
                NumberOfLeaves = 4,
                MinimumExampleCountPerLeaf = 1,
            }));
        return pipeline.Fit(data);
    }

    /// <summary>Fits LightGBM binary classification.</summary>
    /// <typeparam name="TRow">Input row type.</typeparam>
    /// <param name="rows">Training rows.</param>
    /// <param name="labelColumn">Label column name.</param>
    /// <param name="featureColumns">Feature column names to concatenate.</param>
    /// <returns>Fitted transformer.</returns>
    public ITransformer FitLightGbmBinary<TRow>(
        IEnumerable<TRow> rows,
        string labelColumn,
        params string[] featureColumns)
        where TRow : class
    {
        var data = Load(rows);
        var pipeline = FeaturePipeline(featureColumns)
            .Append(_mlContext.BinaryClassification.Trainers.LightGbm(
                labelColumnName: labelColumn,
                featureColumnName: "Features"));
        return pipeline.Fit(data);
    }

    /// <summary>Fits LightGBM regression.</summary>
    /// <typeparam name="TRow">Input row type.</typeparam>
    /// <param name="rows">Training rows.</param>
    /// <param name="labelColumn">Label column name.</param>
    /// <param name="featureColumns">Feature column names to concatenate.</param>
    /// <returns>Fitted transformer.</returns>
    public ITransformer FitLightGbmRegression<TRow>(
        IEnumerable<TRow> rows,
        string labelColumn,
        params string[] featureColumns)
        where TRow : class
    {
        var data = Load(rows);
        var pipeline = FeaturePipeline(featureColumns)
            .Append(_mlContext.Regression.Trainers.LightGbm(
                labelColumnName: labelColumn,
                featureColumnName: "Features"));
        return pipeline.Fit(data);
    }

    /// <summary>Fits SDCA binary logistic regression.</summary>
    /// <typeparam name="TRow">Input row type.</typeparam>
    /// <param name="rows">Training rows.</param>
    /// <param name="labelColumn">Label column name.</param>
    /// <param name="featureColumns">Feature column names to concatenate.</param>
    /// <returns>Fitted transformer.</returns>
    public ITransformer FitSdcaBinary<TRow>(
        IEnumerable<TRow> rows,
        string labelColumn,
        params string[] featureColumns)
        where TRow : class
    {
        var data = Load(rows);
        var pipeline = FeaturePipeline(featureColumns)
            .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                new SdcaLogisticRegressionBinaryTrainer.Options
                {
                    LabelColumnName = labelColumn,
                    FeatureColumnName = "Features",
                    MaximumNumberOfIterations = 1,
                }));
        return pipeline.Fit(data);
    }

    /// <summary>Fits L-BFGS Poisson regression.</summary>
    /// <typeparam name="TRow">Input row type.</typeparam>
    /// <param name="rows">Training rows.</param>
    /// <param name="labelColumn">Label column name.</param>
    /// <param name="featureColumns">Feature column names to concatenate.</param>
    /// <returns>Fitted transformer.</returns>
    public ITransformer FitLbfgsPoissonRegression<TRow>(
        IEnumerable<TRow> rows,
        string labelColumn,
        params string[] featureColumns)
        where TRow : class
    {
        var data = Load(rows);
        var pipeline = FeaturePipeline(featureColumns)
            .Append(_mlContext.Regression.Trainers.LbfgsPoissonRegression(
                labelColumnName: labelColumn,
                featureColumnName: "Features"));
        return pipeline.Fit(data);
    }

    private IDataView Load<TRow>(IEnumerable<TRow> rows) where TRow : class
    {
        ArgumentNullException.ThrowIfNull(rows);
        return _mlContext.Data.LoadFromEnumerable(rows);
    }

    private IEstimator<ITransformer> FeaturePipeline(string[] featureColumns)
    {
        if (featureColumns is null || featureColumns.Length == 0)
            throw new ArgumentException("At least one feature column is required.", nameof(featureColumns));

        return _mlContext.Transforms.Concatenate("Features", featureColumns);
    }
}
