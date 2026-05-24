<!-- novolis-package-index:start -->
> **GitHub Packages shows this repository README on every package page** (upstream limitation).
> Open the **package README** for install and quick start — embedded in each .nupkg and linked below.

## Published packages

| Package | Install | Package README |
|---------|---------|----------------|
| `Novolis.MachineLearning.AutoMl` | `dotnet add package Novolis.MachineLearning.AutoMl` | [README](https://github.com/Novolis-Platform/novolis-machinelearning/blob/main/src/Novolis.MachineLearning.AutoMl/README.md) |
| `Novolis.MachineLearning.Core` | `dotnet add package Novolis.MachineLearning.Core` | [README](https://github.com/Novolis-Platform/novolis-machinelearning/blob/main/src/Novolis.MachineLearning.Core/README.md) |
| `Novolis.MachineLearning.Neural` | `dotnet add package Novolis.MachineLearning.Neural` | [README](https://github.com/Novolis-Platform/novolis-machinelearning/blob/main/src/Novolis.MachineLearning.Neural/README.md) |
| `Novolis.MachineLearning.Neural.Abstractions` | `dotnet add package Novolis.MachineLearning.Neural.Abstractions` | [README](https://github.com/Novolis-Platform/novolis-machinelearning/blob/main/src/Novolis.MachineLearning.Neural.Abstractions/README.md) |

For NuGet.org and Visual Studio, the **embedded** README.md inside each package is authoritative.

<!-- novolis-package-index:end -->

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
dotnet run --project tests/Novolis.MachineLearning.Unit
```

Neural tests use TUnit 1.x as executables (not `dotnet test` on .NET 10 SDK until CI adapter is updated).

## Publishing

NuGet packages ship from this public repo. The Frank.ML source repo may stay private; apps and experiments continue there.

