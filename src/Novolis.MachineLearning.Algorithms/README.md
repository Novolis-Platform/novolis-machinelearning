<!-- novolis-pkg-brand:start -->
<p align="center">
  <a href="https://github.com/Novolis-Platform/novolis-machinelearning">
    <img src="https://raw.githubusercontent.com/Novolis-Platform/.github/main/brand/logo-icon.svg" width="72" alt="Novolis"/>
  </a>
</p>
<!-- novolis-pkg-brand:end -->

# Novolis.MachineLearning.Algorithms

Classic ML.NET trainers and satellite packages: Naive Bayes, FastTree, LightGBM, SDCA, time series, recommenders, image analytics, ONNX, and tokenizers.

## Install

```bash
dotnet add package Novolis.MachineLearning.Algorithms
```

**Prerequisites:** [.NET 10 SDK](https://dotnet.microsoft.com/download) (`net10.0`).

## Quick start

```csharp
using Novolis.MachineLearning.Algorithms;

var trainers = new ClassicTrainers();
var model = trainers.FitNaiveBayes(rows, labelColumn: nameof(Row.Label), featureColumns: ["F1", "F2"]);
var tree = trainers.FitFastTreeBinary(rows, labelColumn: nameof(Row.Label), featureColumns: ["F1", "F2"]);
```

Brought-in NuGets: `Microsoft.ML`, `Microsoft.ML.FastTree`, `Microsoft.ML.LightGbm`, `Microsoft.ML.TimeSeries`, `Microsoft.ML.Recommender`, `Microsoft.ML.ImageAnalytics`, `Microsoft.ML.OnnxTransformer`, `Microsoft.ML.Tokenizers`.

## Related packages

| Package | When to use |
|---------|-------------|
| `Novolis.MachineLearning.AutoMl` | AutoML experiment search |
| `Novolis.MachineLearning.Dump` | Persist models as dump C# / zip |

## Support

Pre-release (`2026.1.*` on GitHub Packages).
