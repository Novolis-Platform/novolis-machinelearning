# novolis-machinelearning

Novolis machine learning packages extracted from [Frank.ML](https://github.com/frankhaugen/Frank.ML).

## Packages

| Package | Description |
|---------|-------------|
| `Novolis.MachineLearning.Core` | IO and data path helpers |
| `Novolis.MachineLearning.Neural.Abstractions` | Dense network contracts and snapshots |
| `Novolis.MachineLearning.Neural` | Training, mutation, JSON persistence |
| `Novolis.MachineLearning.AutoMl` | ML.NET AutoML `ModelSelector` and metrics formatting |

Headless racing simulation: [`Novolis.Simulation.Racing`](../novolis-simulation) in `novolis-simulation`. Evolution demo (trainer + neural controller): [novolis-dogfooding/apps/NeuralRacing](../novolis-dogfooding/apps/NeuralRacing).

Apps, presentation (Avalonia/Spectre), Aspire host, and legacy sample datasets remain in the private Frank.ML repository.

## Build

```bash
dotnet build Novolis.MachineLearning.slnx
dotnet run --project tests/Novolis.MachineLearning.Core.Tests
dotnet run --project tests/Novolis.MachineLearning.Neural.Tests
```

Neural tests use TUnit 1.x as executables (not `dotnet test` on .NET 10 SDK until CI adapter is updated).

## Publishing

NuGet packages ship from this public repo. The Frank.ML source repo may stay private; apps and experiments continue there.
