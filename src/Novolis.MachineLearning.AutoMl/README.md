<!-- novolis-pkg-brand:start -->
<p align="center">
  <a href="https://github.com/Novolis-Platform/novolis-machinelearning">
    <img src="https://raw.githubusercontent.com/Novolis-Platform/.github/main/brand/logo-icon.svg" width="72" alt="Novolis"/>
  </a>
</p>
<!-- novolis-pkg-brand:end -->

# Novolis.MachineLearning.AutoMl

ML.NET AutoML experiment helpers with friendly metric formatting. Also references FastTree, LightGBM, Recommender, and TimeSeries so AutoML trials can use those trainers.

## Install

```bash
dotnet add package Novolis.MachineLearning.AutoMl
```

**Prerequisites:** [.NET 10 SDK](https://dotnet.microsoft.com/download) (`net10.0`).

## Quick start

```csharp
using Microsoft.ML.AutoML;
using Novolis.MachineLearning.AutoMl;

var selector = new ModelSelector<HouseRow>();
ExperimentResult<RegressionMetrics> result = selector.RunRegressionExperiment(
    rows,
    labelColumn: nameof(HouseRow.Price),
    experimentTimeInSeconds: 60);

Console.WriteLine(result.BestRun.ValidationMetrics.ToFriendlyString());
```

For custom neural policies, combine with `Novolis.MachineLearning.Neural` instead of AutoML.

## Related packages

| Package | When to use |
|---------|-------------|
| `Novolis.MachineLearning.Core` | Dataset paths on disk |
| `Novolis.MachineLearning.Neural` | Hand-rolled networks |

## More documentation

- [Getting started](https://github.com/Novolis-Platform/novolis-machinelearning/blob/main/docs/getting-started.md)
- [Design](https://github.com/Novolis-Platform/novolis-machinelearning/blob/main/docs/design.md)

## Support

Pre-release (`2026.1.*` on GitHub Packages).

