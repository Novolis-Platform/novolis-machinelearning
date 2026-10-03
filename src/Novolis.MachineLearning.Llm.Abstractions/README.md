<!-- novolis-pkg-brand:start -->
[![Novolis](https://raw.githubusercontent.com/Novolis-Platform/.github/main/brand/logo-icon.png)](https://novolis-platform.github.io/.github/novolis-machinelearning/)

[Novolis](https://github.com/Novolis-Platform) · [Docs](https://novolis-platform.github.io/.github/novolis-machinelearning/) · [Source](https://github.com/Novolis-Platform/novolis-machinelearning)
<!-- novolis-pkg-brand:end -->

# Novolis.MachineLearning.Llm.Abstractions

Provider-neutral contracts for language-model training runs.

## Install

```bash
dotnet add package Novolis.MachineLearning.Llm.Abstractions
```

**Prerequisites:** [.NET 10 SDK](https://dotnet.microsoft.com/download) (`net10.0`).

## Quick start

Use `ILanguageModelTrainer` with provider packages that support training.

```csharp
ILanguageModelTrainer trainer = /* provider adapter */;
LanguageModelTrainingResult result = await trainer.TrainTextAsync(
[
    "alpha beta gamma",
    "alpha beta delta",
],
new LanguageModelTrainingOptions { TotalSteps = 1 });
```
