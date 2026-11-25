using Onnxify;

namespace Onnxify.Compiler.Operators;

[CompilerTorchOperator("aten::sinc")]
[CompilerTorchOperator("aten::special_sinc")]
internal sealed class SincOperator() : TorchSharpCompositeOperator("Sinc", "com.onnxify.compiler", "torch.sinc", "Tensor.sinc")
{
    public override CompilerOperationCapability Capability => CompilerOperationCapability.Bidirectional;

    public override string PrintTorchSharp(CompilerOnnxStep operation)
    {
        return $"torch.sinc({CompilerCSharpNaming.Identifier(operation.Node.Inputs[0].Name)})";
    }

    public override bool TryScanOnnxSubgraph(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation,
        out int consumedNodeCount)
    {
        operation = null;
        consumedNodeCount = 0;
        if (TryScanCastLikePattern(context, out operation, out consumedNodeCount))
        {
            return true;
        }

        if (!context.TryGetNode<Onnxify.Mul>(0, out var multiply)
            || !context.TryGetNode<Onnxify.Sin>(1, out var sine)
            || !context.TryGetNode<Onnxify.Div>(2, out var divide)
            || !context.TryGetNode<Onnxify.Equal>(3, out var equal)
            || !context.TryGetNode<Onnxify.Where>(4, out var where)
            || multiply is null || sine is null || divide is null || equal is null || where is null
            || !string.Equals(multiply.C.Name, sine.Input.Name, StringComparison.Ordinal)
            || !string.Equals(sine.Output.Name, divide.A.Name, StringComparison.Ordinal)
            || !string.Equals(multiply.C.Name, divide.B.Name, StringComparison.Ordinal)
            || !context.CanCollapse(5))
        {
            return false;
        }

        var inputName = string.Equals(multiply.A.Name, equal.A.Name, StringComparison.Ordinal) ? multiply.A.Name
            : string.Equals(multiply.B.Name, equal.A.Name, StringComparison.Ordinal) ? multiply.B.Name
            : null;
        var piName = inputName == multiply.A.Name ? multiply.B.Name
            : inputName == multiply.B.Name ? multiply.A.Name
            : null;
        var zeroName = equal.A.Name == inputName ? equal.B.Name
            : equal.B.Name == inputName ? equal.A.Name
            : null;
        if (inputName is null
            || piName is null
            || zeroName is null
            || !IsPi(context, piName)
            || !context.IsScalarConstant(zeroName, 0d)
            || !context.IsScalarConstant(where.X.Name, 1d)
            || !string.Equals(equal.C.Name, where.Condition.Name, StringComparison.Ordinal)
            || !string.Equals(divide.C.Name, where.Y.Name, StringComparison.Ordinal))
        {
            return false;
        }

        operation = context.CreateCompositeStep(this, [inputName], where.Output.Name, 5);
        consumedNodeCount = 5;
        return true;
    }

    public override void PrintOnnx(OnnxGraph graph, CompilerOnnxStep operation)
    {
        var zero = AddFloatInitializer(graph, operation, "zero", 0d);
        var one = AddFloatInitializer(graph, operation, "one", 1d);
        var pi = AddFloatInitializer(graph, operation, "pi", Math.PI);
        var piInput = new OnnxEdge($"{operation.Name}__pi_input");
        var sine = new OnnxEdge($"{operation.Name}__sin");
        var ratio = new OnnxEdge($"{operation.Name}__ratio");
        var condition = new OnnxEdge($"{operation.Name}__is_zero");
        graph.AddNode(new Onnxify.Mul($"{operation.Name}__multiply", new Onnxify.MulInputOutputOptions
        {
            A = Input(operation), B = pi, C = piInput,
        }));
        graph.AddNode(new Onnxify.Sin($"{operation.Name}__sine", new Onnxify.SinInputOutputOptions
        {
            Input = piInput, Output = sine,
        }));
        graph.AddNode(new Onnxify.Div($"{operation.Name}__divide", new Onnxify.DivInputOutputOptions
        {
            A = sine, B = piInput, C = ratio,
        }));
        graph.AddNode(new Onnxify.Equal($"{operation.Name}__equal", new Onnxify.EqualInputOutputOptions
        {
            A = Input(operation), B = zero, C = condition,
        }));
        graph.AddNode(new Onnxify.Where(operation.Name, new Onnxify.WhereInputOutputOptions
        {
            Condition = condition, X = one, Y = ratio, Output = Output(operation),
        }));
    }

    private bool TryScanCastLikePattern(
        CompilerOnnxSubgraphScanContext context,
        out CompilerOnnxStep? operation,
        out int consumedNodeCount)
    {
        operation = null;
        consumedNodeCount = 0;
        var graph = context.Graph;
        var start = context.StartIndex;
        if (start < 0 || start + 8 > graph.Nodes.Count)
        {
            return false;
        }

        var window = graph.Nodes.Skip(start).Take(8).ToArray();
        if (window.Count(static node => node.OpType == "CastLike") != 3
            || window.Count(static node => node.OpType == "Mul") != 1
            || window.Count(static node => node.OpType == "Sin") != 1
            || window.Count(static node => node.OpType == "Div") != 1
            || window.Count(static node => node.OpType == "Equal") != 1
            || window.Count(static node => node.OpType == "Where") != 1)
        {
            return false;
        }

        var castLikes = window.Where(static node => string.Equals(node.OpType, "CastLike", StringComparison.Ordinal)).ToArray();
        var multiply = window.OfType<Onnxify.Mul>().SingleOrDefault();
        var sine = window.OfType<Onnxify.Sin>().SingleOrDefault();
        var divide = window.OfType<Onnxify.Div>().SingleOrDefault();
        var equal = window.OfType<Onnxify.Equal>().SingleOrDefault();
        var where = window.OfType<Onnxify.Where>().SingleOrDefault();
        if (castLikes.Length != 3
            || multiply is null || sine is null || divide is null || equal is null || where is null
            || window.Count(static node => node.OpType is "CastLike" or "Mul" or "Sin" or "Div" or "Equal" or "Where") != 8
            || castLikes.Any(static node => !IsCastLike(node)))
        {
            return false;
        }

        var piCast = castLikes.SingleOrDefault(node => IsPi(context, node.Inputs[0].Name));
        var zeroCast = castLikes.SingleOrDefault(node => context.IsScalarConstant(node.Inputs[0].Name, 0d));
        var oneCast = castLikes.SingleOrDefault(node => context.IsScalarConstant(node.Inputs[0].Name, 1d));
        if (piCast is null || zeroCast is null || oneCast is null)
        {
            return false;
        }

        var inputName = piCast.Inputs[1].Name;
        if (!string.Equals(zeroCast.Inputs[1].Name, inputName, StringComparison.Ordinal)
            || !string.Equals(oneCast.Inputs[1].Name, inputName, StringComparison.Ordinal)
            || !HasMulInputs(multiply, inputName, piCast.Outputs[0].Name)
            || !string.Equals(multiply.C.Name, sine.Input.Name, StringComparison.Ordinal)
            || !string.Equals(sine.Output.Name, divide.A.Name, StringComparison.Ordinal)
            || !string.Equals(multiply.C.Name, divide.B.Name, StringComparison.Ordinal)
            || !HasEqualInputs(equal, inputName, zeroCast.Outputs[0].Name)
            || !string.Equals(equal.C.Name, where.Condition.Name, StringComparison.Ordinal)
            || !string.Equals(oneCast.Outputs[0].Name, where.X.Name, StringComparison.Ordinal)
            || !string.Equals(divide.C.Name, where.Y.Name, StringComparison.Ordinal)
            || !ReferenceEquals(window[window.Length - 1], where)
            || !context.CanCollapse(8))
        {
            return false;
        }

        var finalNode = where;
        var name = string.IsNullOrWhiteSpace(finalNode.Name) ? $"sinc_{start}" : finalNode.Name;
        operation = CreateNode(name, [new CompilerValueReference(inputName)],
            [new CompilerValueReference(where.Output.Name)], span: context.Span);
        consumedNodeCount = 8;
        return true;
    }

    private static bool IsCastLike(OnnxNode node)
    {
        return string.IsNullOrEmpty(node.Domain)
            && node.OpType == "CastLike"
            && node.Inputs.Count == 2
            && node.Outputs.Count == 1
            && node.Attributes.All(static attribute =>
                attribute.Name == "saturate" && attribute.GetValue() is long value && value == 1);
    }

    private static bool HasMulInputs(Onnxify.Mul node, string inputName, string scalarName)
    {
        return node.A.Name == inputName && node.B.Name == scalarName
            || node.B.Name == inputName && node.A.Name == scalarName;
    }

    private static bool HasEqualInputs(Onnxify.Equal node, string inputName, string scalarName)
    {
        return node.A.Name == inputName && node.B.Name == scalarName
            || node.B.Name == inputName && node.A.Name == scalarName;
    }

    private static bool IsPi(CompilerOnnxSubgraphScanContext context, string name)
    {
        return context.IsScalarConstant(name, Math.PI)
            || context.IsScalarConstant(name, (double)(float)Math.PI);
    }
}
