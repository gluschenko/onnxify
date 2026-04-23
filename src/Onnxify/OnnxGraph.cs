using Onnx;
using Onnxify.Data;
using Onnxify.Helpers;

namespace Onnxify;

/// <summary>
/// Provides the editable graph surface for ONNX values, initializers, loose edges, and operator nodes.
/// </summary>
/// <remarks>
/// The graph keeps ONNX namespaces explicit. Inputs, outputs, value-info placeholders, initializers, loose edges, and nodes are stored separately but are resolved by the same graph-local names when wiring node inputs and outputs.
/// </remarks>
public class OnnxGraph
{
    /// <summary>
    /// Gets or sets the graph name written to ONNX metadata; assigning <see langword="null"/> stores an empty name.
    /// </summary>
    public string Name
    {
        get => _name;
        set => _name = value ?? string.Empty;
    }

    /// <summary>
    /// Gets graph inputs that callers are expected to feed at inference time.
    /// </summary>
    public IReadOnlyList<OnnxValue> Inputs => _inputs;

    /// <summary>
    /// Gets graph outputs that runtimes expose to callers after execution.
    /// </summary>
    public IReadOnlyList<OnnxValue> Outputs => _outputs;

    /// <summary>
    /// Gets constant tensors stored in the model graph, commonly used for weights, biases, and literal constants.
    /// </summary>
    public IReadOnlyList<OnnxTensor> Initializers => _initializers;

    /// <summary>
    /// Gets ONNX value-info entries that describe intermediate tensors without making them model inputs or outputs.
    /// </summary>
    public IReadOnlyList<OnnxValue> Placeholders => _placeholders;

    /// <summary>
    /// Gets operator nodes in graph order.
    /// </summary>
    public IReadOnlyList<OnnxNode> Nodes => _nodes;

    /// <summary>
    /// Gets a domain accessor used by generated wrappers for ONNX ML-domain operators.
    /// </summary>
    public MLDomain ML => new(this);

    /// <summary>
    /// Gets a domain accessor used by generated wrappers for Microsoft ONNX Runtime extension operators.
    /// </summary>
    public MicrosoftDomain Microsoft => new(this);

    private readonly GraphProto _graph;
    private readonly OnnxModelBaseOptions _options;
    private string _name = string.Empty;
    private readonly LazyDictionary<string, OnnxValue> _inputs = new(x => x.Name, StringComparer.Ordinal);
    private readonly LazyDictionary<string, OnnxValue> _outputs = new(x => x.Name, StringComparer.Ordinal);
    private readonly LazyDictionary<string, OnnxTensor> _initializers = new(x => x.Name, StringComparer.Ordinal);
    private readonly LazyDictionary<string, OnnxValue> _placeholders = new(x => x.Name, StringComparer.Ordinal);
    private readonly LazyDictionary<string, OnnxNode> _nodes = new(x => x.Name, StringComparer.Ordinal);
    private readonly LazyDictionary<string, OnnxEdge> _edges = new(x => x.Name, StringComparer.Ordinal);

    internal OnnxGraph(GraphProto graph, OnnxModelBaseOptions options)
    {
        _graph = graph;
        _options = options;

        foreach (var tensor in graph.Initializer)
        {
            _initializers.Add(OnnxHelper.FromProto(tensor, options));
        }

        foreach (var value in graph.ValueInfo)
        {
            _placeholders.Add(OnnxValue.FromProto(value));
        }

        foreach (var input in graph.Input)
        {
            _inputs.Add(OnnxValue.FromProto(input));
        }

        foreach (var output in graph.Output)
        {
            _outputs.Add(OnnxValue.FromProto(output));
        }

        foreach (var node in graph.Node)
        {
            foreach (var x in node.Input.Concat(node.Output))
            {
                if (GetValue(x) is null)
                {
                    _edges.Add(new OnnxEdge(x));
                }
            }
        }

        foreach (var node in graph.Node)
        {
            _nodes.Add(OnnxNode.FromProto(node, this));
        }

        Name = graph.Name;
    }

    internal OnnxModelBaseOptions GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Finds a node by its exact graph-local name.
    /// </summary>
    /// <param name="name">Node name to look up.</param>
    /// <returns>The matching node, or <see langword="null"/> when the graph does not contain one.</returns>
    public OnnxNode? GetNode(string name)
    {
        if (_nodes.TryGetValue(name, out var result))
        {
            return result;
        }

        return null;
    }

    /// <summary>
    /// Resolves a wire name to the graph object currently representing it.
    /// </summary>
    /// <param name="name">Input, output, value-info, initializer, or loose edge name.</param>
    /// <returns>The matching edge-like object, or <see langword="null"/> when the name is unknown.</returns>
    /// <remarks>
    /// This is the safest way to reuse existing ONNX wires when adding nodes to a loaded graph because it searches all value namespaces used by node inputs and outputs.
    /// </remarks>
    public IOnnxGraphEdge? GetValue(string name)
    {
        if (_inputs.TryGetValue(name, out var input))
        {
            return input;
        }

        if (_outputs.TryGetValue(name, out var output))
        {
            return output;
        }

        if (_placeholders.TryGetValue(name, out var value))
        {
            return value;
        }

        if (_initializers.TryGetValue(name, out var tensor))
        {
            return tensor;
        }

        if (_edges.TryGetValue(name, out var edge))
        {
            return edge;
        }

        return null;
    }

    /// <summary>
    /// Returns <paramref name="prefix"/> when it is unused, otherwise appends a numeric suffix that is free in both node and value namespaces.
    /// </summary>
    /// <param name="prefix">Preferred name or prefix for generated graph members.</param>
    /// <returns>A graph-local name that will not collide with an existing node, value, initializer, or loose edge.</returns>
    public string NextName(string prefix)
    {
        var node = GetNode(prefix);
        var value = GetValue(prefix);

        if (node is not null || value is not null)
        {
            var counter = 0;

            while (true)
            {
                var candidate = $"{prefix}_{counter}";

                node = GetNode(candidate);
                value = GetValue(candidate);

                if (node is null && value is null)
                {
                    return candidate;
                }

                counter++;
            }
        }

        return prefix;
    }

    /// <summary>
    /// Adds an initializer tensor that will be stored with the graph.
    /// </summary>
    /// <typeparam name="T">CLR element type mapped to an ONNX tensor element type.</typeparam>
    /// <param name="name">Tensor name used by node inputs.</param>
    /// <param name="shape">Tensor dimensions in ONNX order.</param>
    /// <param name="value">Flat tensor data in row-major order.</param>
    /// <returns>The initializer object that can be wired into nodes.</returns>
    /// <exception cref="InvalidOperationException">Thrown when another initializer already uses <paramref name="name"/>.</exception>
    public OnnxTensor AddTensor<T>(
        string name,
        long[] shape,
        T[] value
    )
    {
        if (_initializers.Contains(name))
        {
            throw new InvalidOperationException($"Tensor '{name}' is already added into graph");
        }

        var tensor = new OnnxTensor<T>(
            name: name,
            dataLocation: OnnxTensor.TensorDataLocation.Default,
            shape: shape,
            value: value,
            null
        );

        _initializers.Add(tensor);
        return tensor;
    }

    /// <summary>
    /// Adds a graph input with explicit ONNX type information.
    /// </summary>
    /// <typeparam name="T">Concrete value-type descriptor, typically <see cref="OnnxTensorType"/>.</typeparam>
    /// <param name="name">Input name expected by runtimes and callers.</param>
    /// <param name="type">ONNX type descriptor for the input.</param>
    /// <returns>The input value that can be connected to node inputs.</returns>
    /// <exception cref="InvalidOperationException">Thrown when another graph input already uses <paramref name="name"/>.</exception>
    public OnnxValue AddInput<T>(string name, T type) where T : OnnxValueType
    {
        if (_inputs.Contains(name))
        {
            throw new InvalidOperationException($"Value '{name}' is already added into graph");
        }

        var placeholder = new OnnxValue<T>(name, type, null);
        _inputs.Add(placeholder);
        return placeholder;
    }

    /// <summary>
    /// Adds a graph output with explicit ONNX type information.
    /// </summary>
    /// <typeparam name="T">Concrete value-type descriptor, typically <see cref="OnnxTensorType"/>.</typeparam>
    /// <param name="name">Output name exposed by runtimes.</param>
    /// <param name="type">ONNX type descriptor for the output.</param>
    /// <returns>The output value that can be connected to node outputs.</returns>
    /// <exception cref="InvalidOperationException">Thrown when another graph output already uses <paramref name="name"/>.</exception>
    public OnnxValue AddOutput<T>(string name, T type) where T : OnnxValueType
    {
        if (_outputs.Contains(name))
        {
            throw new InvalidOperationException($"Value '{name}' is already added into graph");
        }

        var placeholder = new OnnxValue<T>(name, type, null);
        _outputs.Add(placeholder);
        return placeholder;
    }

    /// <summary>
    /// Adds intermediate value-info for a tensor that should be described but not exposed as a model input or output.
    /// </summary>
    /// <typeparam name="T">Concrete value-type descriptor, typically <see cref="OnnxTensorType"/>.</typeparam>
    /// <param name="name">Intermediate wire name.</param>
    /// <param name="type">ONNX type descriptor for the intermediate value.</param>
    /// <returns>The placeholder value that can be connected to nodes.</returns>
    /// <exception cref="InvalidOperationException">Thrown when another placeholder already uses <paramref name="name"/>.</exception>
    public OnnxValue AddValue<T>(string name, T type) where T : OnnxValueType
    {
        if (_placeholders.Contains(name))
        {
            throw new InvalidOperationException($"Value '{name}' is already added into graph");
        }

        var placeholder = new OnnxValue<T>(name, type, null);
        _placeholders.Add(placeholder);
        return placeholder;
    }

    /// <summary>
    /// Adds a named wire without value-info metadata.
    /// </summary>
    /// <param name="name">Wire name to use in node input or output lists.</param>
    /// <returns>An edge object suitable for wiring nodes before type metadata is known.</returns>
    /// <exception cref="InvalidOperationException">Thrown when another loose edge already uses <paramref name="name"/>.</exception>
    public OnnxEdge AddEdge(string name)
    {
        if (_edges.Contains(name))
        {
            throw new InvalidOperationException($"Edge '{name}' is already added into graph");
        }

        var edge = new OnnxEdge(name);
        _edges.Add(edge);
        return edge;
    }

    /// <summary>
    /// Creates and adds a generic ONNX node.
    /// </summary>
    /// <param name="name">Graph-local node name.</param>
    /// <param name="opType">Operator type, such as <c>Conv</c>, <c>MatMul</c>, or <c>Relu</c>.</param>
    /// <param name="domain">Operator domain. Use an empty string for standard ONNX operators.</param>
    /// <param name="docString">Optional ONNX node documentation string.</param>
    /// <param name="inputs">Input wires in schema order.</param>
    /// <param name="outputs">Output wires in schema order.</param>
    /// <param name="attributes">Operator attributes keyed by ONNX attribute name.</param>
    /// <returns>The added node.</returns>
    public OnnxNode AddNode(
        string name,
        string opType,
        string domain,
        string docString,
        IEnumerable<IOnnxGraphEdge> inputs,
        IEnumerable<IOnnxGraphEdge> outputs,
        IEnumerable<OnnxAttribute> attributes
    )
    {
        var node = new OnnxNode(
            name: name,
            opType: opType,
            domain: domain,
            docString: docString,
            inputs: inputs,
            outputs: outputs,
            attributes: attributes,
            proto: null
        );

        return AddNode(node);
    }

    /// <summary>
    /// Adds a prepared node instance to the graph.
    /// </summary>
    /// <param name="node">Node to add. Generated operator wrapper instances can be passed here.</param>
    /// <returns>The same node instance, after it has been registered with the graph.</returns>
    /// <exception cref="InvalidOperationException">Thrown when another node already uses <paramref name="node"/>'s name.</exception>
    public OnnxNode AddNode(OnnxNode node)
    {
        if (_nodes.Contains(node.Name))
        {
            throw new InvalidOperationException($"Node '{node.Name}' is already added into graph");
        }

        _nodes.Add(node);
        return node;
    }

    internal GraphProto ToProto()
    {
        var newGraph = _graph.Clone();
        newGraph.Name = Name;

        newGraph.Initializer.Set(_initializers.Select(x => x.ToProto()));
        newGraph.ValueInfo.Set(_placeholders.Select(x => x.ToProto()));
        newGraph.Input.Set(_inputs.Select(x => x.ToProto()));
        newGraph.Output.Set(_outputs.Select(x => x.ToProto()));
        newGraph.Node.Set(_nodes.Select(x => x.ToProto()));

        return newGraph;
    }

    /// <summary>
    /// Returns a diagnostic view of graph members and their wiring.
    /// </summary>
    /// <returns>A multiline string intended for inspection, not for stable serialization.</returns>
    public override string ToString()
    {
        var name = string.IsNullOrWhiteSpace(Name) ? "<unnamed>" : Name;
        return $"""
            OnnxGraph(
                Name={name},
                Inputs={FormatCollection(Inputs).Indent(1)},
                Outputs={FormatCollection(Outputs).Indent(1)},
                Initializers={FormatCollection(Initializers).Indent(1)},
                Values={FormatCollection(Placeholders).Indent(1)},
                Nodes={FormatCollection(Nodes).Indent(1)}
            )
            """;
    }

    private static string FormatCollection<T>(IEnumerable<T> values)
    {
        return $"""
            [
                {string.Join(",\n", values).Indent(1)}
            ]
            """;
    }
}

/// <summary>
/// Entry point for generated wrappers in the ONNX ML operator domain.
/// </summary>
public readonly struct MLDomain(OnnxGraph graph)
{
    /// <summary>
    /// Gets the graph that generated ML-domain wrapper calls will mutate.
    /// </summary>
    public readonly OnnxGraph Graph = graph;
}

/// <summary>
/// Entry point for generated wrappers in Microsoft ONNX Runtime extension domains.
/// </summary>
public readonly struct MicrosoftDomain(OnnxGraph graph)
{
    /// <summary>
    /// Gets the graph that generated Microsoft-domain wrapper calls will mutate.
    /// </summary>
    public readonly OnnxGraph Graph = graph;

    /// <summary>
    /// Gets wrappers for the <c>com.microsoft.internal</c> domain.
    /// </summary>
    public MicrosoftInternalDomain Internal => new(Graph);

    /// <summary>
    /// Gets wrappers for Microsoft NHWC operators exposed from this domain accessor.
    /// </summary>
    public MicrosoftInternalNHWCDomain NHWC => new(Graph);

    /// <summary>
    /// Gets wrappers for Microsoft NCHWc layout operators.
    /// </summary>
    public MicrosoftNCHWcDomain NCHWc => new(Graph);
}

/// <summary>
/// Entry point for generated wrappers in the Microsoft internal operator domain.
/// </summary>
public readonly struct MicrosoftInternalDomain(OnnxGraph graph)
{
    /// <summary>
    /// Gets the graph that generated Microsoft-internal wrapper calls will mutate.
    /// </summary>
    public readonly OnnxGraph Graph = graph;

    /// <summary>
    /// Gets wrappers for Microsoft internal NHWC operators.
    /// </summary>
    public MicrosoftInternalNHWCDomain NHWC => new(Graph);
}

/// <summary>
/// Entry point for generated wrappers for Microsoft internal NHWC-layout operators.
/// </summary>
public readonly struct MicrosoftInternalNHWCDomain(OnnxGraph graph)
{
    /// <summary>
    /// Gets the graph that generated NHWC wrapper calls will mutate.
    /// </summary>
    public readonly OnnxGraph Graph = graph;
}

/// <summary>
/// Entry point for generated wrappers for Microsoft NHWC-layout operators.
/// </summary>
public readonly struct MicrosoftNHWCDomain(OnnxGraph graph)
{
    /// <summary>
    /// Gets the graph that generated NHWC wrapper calls will mutate.
    /// </summary>
    public readonly OnnxGraph Graph = graph;
}

/// <summary>
/// Entry point for generated wrappers for Microsoft NCHWc-layout operators.
/// </summary>
public readonly struct MicrosoftNCHWcDomain(OnnxGraph graph)
{
    /// <summary>
    /// Gets the graph that generated NCHWc wrapper calls will mutate.
    /// </summary>
    public readonly OnnxGraph Graph = graph;
}
