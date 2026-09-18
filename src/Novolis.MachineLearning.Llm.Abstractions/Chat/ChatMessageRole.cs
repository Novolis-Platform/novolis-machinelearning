namespace Novolis.MachineLearning.Llm;

/// <summary>Role of a message in a chat transcript.</summary>
public enum ChatMessageRole
{
    /// <summary>Instructions that steer model behavior.</summary>
    System,

    /// <summary>Input from the caller or end user.</summary>
    User,

    /// <summary>Output from the model or assistant.</summary>
    Assistant,
}
