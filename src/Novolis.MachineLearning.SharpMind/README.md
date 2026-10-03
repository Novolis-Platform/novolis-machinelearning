<!-- novolis-pkg-brand:start -->
[![Novolis](https://raw.githubusercontent.com/Novolis-Platform/.github/main/brand/logo-icon.png)](https://novolis-platform.github.io/.github/novolis-machinelearning/)

[Novolis](https://github.com/Novolis-Platform) · [Docs](https://novolis-platform.github.io/.github/novolis-machinelearning/) · [Source](https://github.com/Novolis-Platform/novolis-machinelearning)
<!-- novolis-pkg-brand:end -->

# Novolis.MachineLearning.SharpMind

Optional SharpMind-backed tiny from-scratch text training adapter.

## Install

```bash
dotnet add package Novolis.MachineLearning.SharpMind
```

**Prerequisites:** [.NET 10 SDK](https://dotnet.microsoft.com/download) (`net10.0`).

## Quick start

```csharp
using Novolis.MachineLearning.SharpMind;

var trainer = new SharpMindTextTrainer();
var result = await trainer.TrainTextAsync(
[
    "alpha beta gamma",
    "alpha beta delta",
    "beta gamma alpha",
],
new SharpMindTextTrainingOptions
{
    TotalSteps = 1,
    BatchSize = 1,
    SequenceLength = 8,
    VocabularySize = 16,
});

Console.WriteLine(result.FinalLoss);
```

This training path is meant for small local experiments and CI smoke tests. Larger LLM training and LoRA workflows should get narrower APIs once their artifact formats and data contracts are settled.
