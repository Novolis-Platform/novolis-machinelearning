<!-- novolis-pkg-brand:start -->
<p align="center">
  <a href="https://github.com/Novolis-Platform/novolis-machinelearning">
    <img src="https://raw.githubusercontent.com/Novolis-Platform/.github/main/brand/logo-icon.svg" width="72" alt="Novolis"/>
  </a>
</p>
<!-- novolis-pkg-brand:end -->

# Novolis.MachineLearning.Algorithms

Classic ML.NET trainers plus a typed Naive Bayes library (`Features<T>`, Gaussian / Bernoulli) and satellite packages: FastTree, LightGBM, SDCA, time series, recommenders, image analytics, ONNX, and tokenizers.

## Install

```bash
dotnet add package Novolis.MachineLearning.Algorithms
```

**Prerequisites:** [.NET 10 SDK](https://dotnet.microsoft.com/download) (`net10.0`).

## Quick start

```csharp
using Novolis.MachineLearning.Algorithms;
using Novolis.MachineLearning.Algorithms.NaiveBayes;

// Typed Features<T> Naive Bayes (pure C#, no ML.NET pipeline)
var trainer = new GaussianNaiveBayesTrainer<double, string>();
var model = trainer.Fit(
[
    new(new Features<double>(5.1, 3.5, 1.4, 0.2), "setosa"),
    new(new Features<double>(6.4, 3.2, 4.5, 1.5), "versicolor"),
]);
var species = model.Predict(new Features<double>(5.0, 3.4, 1.5, 0.2));

// ML.NET classic trainers
var trainers = new ClassicTrainers();
var nb = trainers.FitNaiveBayes(rows, labelColumn: nameof(Row.Label), featureColumns: ["F1", "F2"]);
var tree = trainers.FitFastTreeBinary(rows, labelColumn: nameof(Row.Label), featureColumns: ["F1", "F2"]);
```

`Features<T>` requires unmanaged `IEquatable<T>` elements. Gaussian Naive Bayes further requires `INumber<T>`; Bernoulli Naive Bayes is limited to `Features<bool>`.

Brought-in NuGets: `Microsoft.ML`, `Microsoft.ML.FastTree`, `Microsoft.ML.LightGbm`, `Microsoft.ML.TimeSeries`, `Microsoft.ML.Recommender`, `Microsoft.ML.ImageAnalytics`, `Microsoft.ML.OnnxTransformer`, `Microsoft.ML.Tokenizers`.

## Related packages

| Package | When to use |
|---------|-------------|
| `Novolis.MachineLearning.AutoMl` | AutoML experiment search |
| `Novolis.MachineLearning.Dump` | Persist models as dump C# / zip |

## Support

Pre-release (`2026.1.*` on GitHub Packages).
