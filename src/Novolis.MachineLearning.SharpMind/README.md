<!-- novolis-pkg-brand:start -->
<p align="center">
  <a href="https://github.com/Novolis-Platform/novolis-machinelearning">
    <img src="https://raw.githubusercontent.com/Novolis-Platform/.github/main/brand/logo-icon.svg" width="72" alt="Novolis"/>
  </a>
</p>
<!-- novolis-pkg-brand:end -->

# Novolis.MachineLearning.SharpMind

Optional SharpMind-backed local LLM chat and tiny from-scratch text training adapter.

## Install

```bash
dotnet add package Novolis.MachineLearning.SharpMind
```

**Prerequisites:** [.NET 10 SDK](https://dotnet.microsoft.com/download) (`net10.0`) and a local SharpMind-supported model file such as GGUF or SMM.

## Quick start

```csharp
using Novolis.MachineLearning.Llm;
using Novolis.MachineLearning.SharpMind;

await using var model = SharpMindChatModel.Load(new SharpMindChatModelOptions
{
    ModelPath = @"C:\Models\Qwen3-0.6B-Q8_0.gguf",
    LoadMode = SharpMindModelLoadMode.Streaming,
});

var response = await model.GenerateAsync(
[
    new ChatMessage(ChatMessageRole.System, "Be concise."),
    new ChatMessage(ChatMessageRole.User, "Explain the current simulation state."),
]);
```

## Tiny training smoke

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
