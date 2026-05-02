namespace Onnxify.ML.Stages;

public sealed class ForkResult<TInput, TLeft, TRight>
{
    public required TInput Input { get; init; }

    public required IReadOnlyList<TLeft> Left { get; init; }

    public required IReadOnlyList<TRight> Right { get; init; }
}
