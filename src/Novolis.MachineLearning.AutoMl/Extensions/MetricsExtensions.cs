using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using Microsoft.ML;
using Microsoft.ML.Data;

namespace Novolis.MachineLearning.AutoMl.Extensions;

/// <summary>Human-readable summaries of ML.NET metric objects.</summary>
public static class MetricsExtensions
{
    /// <summary>Formats multiclass classification metrics for console output.</summary>
    public static string ToFriendlyString(this MulticlassClassificationMetrics metrics)
    {
        var stringBuilder = new StringBuilder();
        stringBuilder.AppendLine($"************************************************************");
        stringBuilder.AppendLine($"*    Metrics for multi-class classification model   ");
        stringBuilder.AppendLine($"*-----------------------------------------------------------");
        stringBuilder.AppendLine($"    MacroAccuracy = {metrics.MacroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
        stringBuilder.AppendLine($"    MicroAccuracy = {metrics.MicroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
        stringBuilder.AppendLine($"    LogLoss = {metrics.LogLoss:0.####}, the closer to 0, the better");
        for (int i = 0; i < metrics.PerClassLogLoss.Count; i++)
        {
            stringBuilder.AppendLine($"    LogLoss for class {i + 1} = {metrics.PerClassLogLoss[i]:0.####}, the closer to 0, the better");
        }
        stringBuilder.AppendLine($"************************************************************");

        return stringBuilder.ToString();
    }

    /// <summary>Formats cross-validated multiclass metrics.</summary>
    public static string ToFriendlyString(this IEnumerable<TrainCatalogBase.CrossValidationResult<MulticlassClassificationMetrics>> crossValResults)
    {
        var metricsInMultipleFolds = crossValResults.Select(r => r.Metrics);

        var microAccuracyValues = metricsInMultipleFolds.Select(m => m.MicroAccuracy);
        var microAccuracyAverage = microAccuracyValues.Average();
        var microAccuraciesStdDeviation = CalculateStandardDeviation(microAccuracyValues);
        var microAccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(microAccuracyValues);

        var macroAccuracyValues = metricsInMultipleFolds.Select(m => m.MacroAccuracy);
        var macroAccuracyAverage = macroAccuracyValues.Average();
        var macroAccuraciesStdDeviation = CalculateStandardDeviation(macroAccuracyValues);
        var macroAccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(macroAccuracyValues);

        var logLossValues = metricsInMultipleFolds.Select(m => m.LogLoss);
        var logLossAverage = logLossValues.Average();
        var logLossStdDeviation = CalculateStandardDeviation(logLossValues);
        var logLossConfidenceInterval95 = CalculateConfidenceInterval95(logLossValues);

        var logLossReductionValues = metricsInMultipleFolds.Select(m => m.LogLossReduction);
        var logLossReductionAverage = logLossReductionValues.Average();
        var logLossReductionStdDeviation = CalculateStandardDeviation(logLossReductionValues);
        var logLossReductionConfidenceInterval95 = CalculateConfidenceInterval95(logLossReductionValues);

        var stringBuilder = new StringBuilder();
        stringBuilder.AppendLine($"*************************************************************************************************************");
        stringBuilder.AppendLine($"*       Metrics for Multi-class Classification model      ");
        stringBuilder.AppendLine($"*------------------------------------------------------------------------------------------------------------");
        stringBuilder.AppendLine($"*       Average MicroAccuracy:    {microAccuracyAverage:0.###}  - Standard deviation: ({microAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({microAccuraciesConfidenceInterval95:#.###})");
        stringBuilder.AppendLine($"*       Average MacroAccuracy:    {macroAccuracyAverage:0.###}  - Standard deviation: ({macroAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({macroAccuraciesConfidenceInterval95:#.###})");
        stringBuilder.AppendLine($"*       Average LogLoss:          {logLossAverage:#.###}  - Standard deviation: ({logLossStdDeviation:#.###})  - Confidence Interval 95%: ({logLossConfidenceInterval95:#.###})");
        stringBuilder.AppendLine($"*       Average LogLossReduction: {logLossReductionAverage:#.###}  - Standard deviation: ({logLossReductionStdDeviation:#.###})  - Confidence Interval 95%: ({logLossReductionConfidenceInterval95:#.###})");
        stringBuilder.AppendLine($"*************************************************************************************************************");

        return stringBuilder.ToString();
    }

    /// <summary>Formats regression metrics for console output.</summary>
    public static string ToFriendlyString(this RegressionMetrics metrics)
    {
        var stringBuilder = new StringBuilder();
        stringBuilder.AppendLine($"*************************************************");
        stringBuilder.AppendLine($"*       Metrics for Recommendation model      ");
        stringBuilder.AppendLine($"*------------------------------------------------");
        stringBuilder.AppendLine($"*       LossFn:        {metrics.LossFunction:0.##}");
        stringBuilder.AppendLine($"*       R2 Score:      {metrics.RSquared:0.##}");
        stringBuilder.AppendLine($"*       Absolute loss: {metrics.MeanAbsoluteError:#.##}");
        stringBuilder.AppendLine($"*       Squared loss:  {metrics.MeanSquaredError:#.##}");
        stringBuilder.AppendLine($"*       RMS loss:      {metrics.RootMeanSquaredError:#.##}");
        stringBuilder.AppendLine($"*************************************************");

        return stringBuilder.ToString();
    }

    /// <summary>Formats cross-validated regression metrics.</summary>
    public static string ToFriendlyString(this IEnumerable<TrainCatalogBase.CrossValidationResult<RegressionMetrics>> crossValidationResults)
    {
        var L1 = crossValidationResults.Select(r => r.Metrics.MeanAbsoluteError);
        var L2 = crossValidationResults.Select(r => r.Metrics.MeanSquaredError);
        var RMS = crossValidationResults.Select(r => r.Metrics.RootMeanSquaredError);
        var lossFunction = crossValidationResults.Select(r => r.Metrics.LossFunction);
        var R2 = crossValidationResults.Select(r => r.Metrics.RSquared);

        var stringBuilder = new StringBuilder();
        stringBuilder.AppendLine($"*************************************************************************************************************");
        stringBuilder.AppendLine($"*       Metrics for Recommendation model      ");
        stringBuilder.AppendLine($"*------------------------------------------------------------------------------------------------------------");
        stringBuilder.AppendLine($"*       Average L1 Loss:       {L1.Average():0.###} ");
        stringBuilder.AppendLine($"*       Average L2 Loss:       {L2.Average():0.###}  ");
        stringBuilder.AppendLine($"*       Average RMS:           {RMS.Average():0.###}  ");
        stringBuilder.AppendLine($"*       Average Loss Function: {lossFunction.Average():0.###}  ");
        stringBuilder.AppendLine($"*       Average R-squared:     {R2.Average():0.###}  ");
        stringBuilder.AppendLine($"*************************************************************************************************************");

        return stringBuilder.ToString();
    }

    /// <summary>Sample standard deviation.</summary>
    public static double CalculateStandardDeviation(IEnumerable<double> values)
    {
        double average = values.Average();
        double sumOfSquaresOfDifferences = values.Select(val => (val - average) * (val - average)).Sum();
        double standardDeviation = Math.Sqrt(sumOfSquaresOfDifferences / (values.Count() - 1));
        return standardDeviation;
    }

    /// <summary>Half-width of the 95% confidence interval for the mean.</summary>
    public static double CalculateConfidenceInterval95(IEnumerable<double> values)
    {
        double confidenceInterval95 = 1.96 * CalculateStandardDeviation(values) / Math.Sqrt((values.Count() - 1));
        return confidenceInterval95;
    }
}
