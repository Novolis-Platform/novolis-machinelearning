# novolis-machinelearning

Novolis machine learning foundation packages extracted from [Frank.ML](https://github.com/frankhaugen/Frank.ML) (wave 8 — neural stack only).

## Packages

| Package | Description |
|---------|-------------|
| `Novolis.MachineLearning.Core` | IO and data path helpers |
| `Novolis.MachineLearning.Neural.Abstractions` | Dense network contracts and snapshots |
| `Novolis.MachineLearning.Neural` | Training, mutation, JSON persistence |

AutoML (`Microsoft.ML`) and app/domain code remain in the private Frank.ML repository.

## Build

```bash
dotnet build Novolis.MachineLearning.slnx
dotnet run --project tests/Novolis.MachineLearning.Core.Tests
dotnet run --project tests/Novolis.MachineLearning.Neural.Tests
```

Neural tests use TUnit 1.x as executables (not `dotnet test` on .NET 10 SDK until CI adapter is updated).

## Publishing

NuGet packages ship from this public repo. The Frank.ML source repo may stay private; apps and experiments continue there.
