# Novolis.MachineLearning.Neural.Abstractions

Contracts for neural networks, evaluation, and snapshot persistence.

## Install

```bash
dotnet add package Novolis.MachineLearning.Neural.Abstractions
```

**Prerequisites:** [.NET 10 SDK](https://dotnet.microsoft.com/download) (`net10.0`).

## Quick start

```csharp
using Novolis.MachineLearning.Neural;
using Novolis.MachineLearning.Neural.Persistence;

INeuralNetwork network = /* DenseNetwork or test double */;
ReadOnlyMemory<double> output = network.Forward(inputSpan);
NetworkEvaluation eval = network.Evaluate(inputSpan);

// Persistence contracts: INeuralNetworkRepository, NetworkSnapshot
```

Reference this package for mocks; use `Novolis.MachineLearning.Neural` for the default implementation.

## Related packages

| Package | When to use |
|---------|-------------|
| `Novolis.MachineLearning.Neural` | `DenseNetwork` training and file repository |
| `Novolis.MachineLearning.Core` | Data directory layout |

## More documentation

- [Getting started](https://github.com/Novolis-Platform/novolis-machinelearning/blob/main/docs/getting-started.md)
- [Design](https://github.com/Novolis-Platform/novolis-machinelearning/blob/main/docs/design.md)

## Support

Pre-release (`2026.1.*` on GitHub Packages).
