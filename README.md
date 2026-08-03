<!-- novolis-marketing:start -->
<p align="center">
  <a href="https://github.com/Novolis-Platform">
    <img src="https://raw.githubusercontent.com/Novolis-Platform/.github/main/brand/logo-brand-transparent.svg" width="360" alt="Novolis"/>
  </a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Novolis-Platform/.github/main/brand/banners/novolis-machinelearning.svg" width="100%" alt="novolis-machinelearning"/>
</p>

<p align="center">
  <strong>AutoML and neural helpers</strong><br/>
  Machine learning core, AutoML, and neural utilities for Novolis.
</p>

<p align="center">
  <a href="https://github.com/Novolis-Platform/novolis-machinelearning/actions"><img src="https://img.shields.io/github/actions/workflow/status/Novolis-Platform/novolis-machinelearning/merge.yml?branch=main&label=merge&logo=github" alt="merge"/></a>
  <a href="https://github.com/orgs/Novolis-Platform/packages?repo_name=novolis-machinelearning"><img src="https://img.shields.io/badge/packages-GitHub%20Packages-0a7ea3?logo=nuget" alt="packages"/></a>
  <a href="https://github.com/Novolis-Platform"><img src="https://img.shields.io/badge/org-Novolis--Platform-111827" alt="org"/></a>
</p>

<p align="center">
  <a href="https://nuget.pkg.github.com/Novolis-Platform/index.json"><code>https://nuget.pkg.github.com/Novolis-Platform/index.json</code></a>
  ·
  <a href="https://github.com/Novolis-Platform/.github/blob/main/profile/README.md">Org landing</a>
  ·
  <a href="https://github.com/Novolis-Platform/novolis-governance">Governance</a>
</p>

---
<!-- novolis-marketing:end -->
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

