namespace Onnxify.ML.Stages;

/// <summary>
/// Contains the outputs produced by a <see cref="ForkStage{TInput, TLeft, TRight}"/> for a single input item.
/// </summary>
public sealed class ForkResult<TInput, TLeft, TRight>
{
    /// <summary>
    /// Gets the original input item that was forked.
    /// </summary>
    public required TInput Input { get; init; }

    /// <summary>
    /// Gets all outputs produced by the left child stage.
    /// </summary>
    public required IReadOnlyList<TLeft> Left { get; init; }

    /// <summary>
    /// Gets all outputs produced by the right child stage.
    /// </summary>
    public required IReadOnlyList<TRight> Right { get; init; }
}
