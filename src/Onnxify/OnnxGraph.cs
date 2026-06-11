using Onnx;
using Onnxify.Data;
using Onnxify.Helpers;

namespace Onnxify;

/// <summary>
/// Provides the editable graph surface for ONNX values, initializers, loose edges, and operator nodes.
/// </summary>
/// <remarks>
/// The graph keeps ONNX namespaces explicit. Inputs, outputs, intermediate values, initializers, loose edges, and nodes are stored separately but are resolved by the same graph-local names when wiring node inputs and outputs.
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
    public IReadOnlyList<OnnxValue> Inputs => _values.Where(x => _inputs.Contains(x.Name)).ToList();

    /// <summary>
    /// Gets graph outputs that runtimes expose to callers after execution.
    /// </summary>
    public IReadOnlyList<OnnxValue> Outputs => _values.Where(x => _outputs.Contains(x.Name)).ToList();

    /// <summary>
    /// Gets constant tensors stored in the model graph, commonly used for weights, biases, and literal constants.
    /// </summary>
    public IReadOnlyList<OnnxTensor> Initializers => _initializers;

    /// <summary>
    /// Gets ONNX value-info entries that describe intermediate tensors without making them model inputs or outputs.
    /// </summary>
    public IReadOnlyList<OnnxValue> IntermediateValues => _values.Where(x => !_inputs.Contains(x.Name) && !_outputs.Contains(x.Name)).ToList();

    /// <summary>
    /// Gets operator nodes in graph order.
    /// </summary>
    public IReadOnlyList<OnnxNode> Nodes => _nodes;

    /// <summary>
    /// Gets a domain accessor used by generated wrappers for ONNX ML-domain operators.
    /// 
    /// <para>ai.onnx.ml</para>
    /// </summary>
    public MLDomain ML => new(this);

    /// <summary>
    /// Gets a domain accessor used by generated wrappers for Microsoft ONNX Runtime extension operators.
    /// 
    /// <para>com.microsoft.*</para>
    /// </summary>
    public MicrosoftDomain Microsoft => new(this);

    private readonly GraphProto _graph;
    private readonly OnnxModelBaseOptions _options;
    private string _name = string.Empty;

    private readonly HashSet<string> _inputs = new(StringComparer.Ordinal);
    private readonly HashSet<string> _outputs = new(StringComparer.Ordinal);
    // private readonly LazyDictionary<string, OnnxValue> _inputs = new(x => x.Name, StringComparer.Ordinal);
    // private readonly LazyDictionary<string, OnnxValue> _outputs = new(x => x.Name, StringComparer.Ordinal);
    private readonly LazyDictionary<string, OnnxTensor> _initializers = new(x => x.Name, StringComparer.Ordinal);
    private readonly LazyDictionary<string, OnnxValue> _values = new(x => x.Name, StringComparer.Ordinal);
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
            _values.Add(OnnxValue.FromProto(value));
        }

        foreach (var input in graph.Input)
        {
            var x = OnnxValue.FromProto(input);
            _values.Add(x);
            _inputs.Add(x.Name);
        }

        foreach (var output in graph.Output)
        {
            var x = OnnxValue.FromProto(output);
            _values.Add(x);
            _outputs.Add(x.Name);
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

    internal bool TryGetImportedOpset(string domain, out long version)
    {
        if (_options.OpsetImports is null)
        {
            version = default;
            return false;
        }

        if (!_options.OpsetImports.TryGetValue(domain, out version))
        {
            version = default;
            return false;
        }

        return true;
    }

    /// <summary>
    /// Finds a node by its exact graph-local name.
    /// </summary>
    /// <param name="name">Node name to look up.</param>
    /// <returns>The matching node, or <see langword="null"/> when the graph does not contain one.</returns>
    public OnnxNode? GetNode(string name)
    {
        ArgumentNullException.ThrowIfNull(name);

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
        ArgumentNullException.ThrowIfNull(name);

        if (_values.TryGetValue(name, out var value))
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
    /// Removes an initializer tensor and clears node input or output references to the same graph wire.
    /// </summary>
    /// <param name="name">Initializer name to remove.</param>
    /// <returns><see langword="true"/> when a tensor or node reference was removed.</returns>
    public bool RemoveTensor(string name)
    {
        ArgumentNullException.ThrowIfNull(name);

        var removed = _initializers.Remove(name);
        removed |= _edges.Remove(name);
        removed |= RemoveNodeEdgeReferences(name);
        PruneUnusedLooseEdges();

        return removed;
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

        var value = new OnnxValue<T>(name, type, null);
        AddValue(value);
        _inputs.Add(value.Name);
        return value;
    }

    /// <summary>
    /// Marks an existing graph value as a model input.
    /// </summary>
    /// <param name="value">Previously added graph value.</param>
    /// <returns>The registered graph value.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the value is not registered in the graph.</exception>
    public OnnxValue AddInput(OnnxValue value)
    {
        ArgumentNullException.ThrowIfNull(value);

        var registeredValue = GetRegisteredValue(value.Name);
        _inputs.Add(registeredValue.Name);

        return registeredValue;
    }

    /// <summary>
    /// Removes a graph value from the model input list without deleting the value-info entry itself.
    /// </summary>
    /// <param name="name">Input name to unmark.</param>
    /// <returns><see langword="true"/> when the value was marked as an input.</returns>
    public bool RemoveInput(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        return _inputs.Remove(name);
    }

    /// <summary>
    /// Removes a graph value from the model input list without deleting the value-info entry itself.
    /// </summary>
    /// <param name="value">Input value to unmark.</param>
    /// <returns><see langword="true"/> when the value was marked as an input.</returns>
    public bool RemoveInput(OnnxValue value)
    {
        ArgumentNullException.ThrowIfNull(value);
        return RemoveInput(value.Name);
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

        var value = new OnnxValue<T>(name, type, null);
        AddValue(value);
        _outputs.Add(value.Name);
        return value;
    }

    /// <summary>
    /// Marks an existing graph value as a model output.
    /// </summary>
    /// <param name="value">Previously added graph value.</param>
    /// <returns>The registered graph value.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the value is not registered in the graph.</exception>
    public OnnxValue AddOutput(OnnxValue value)
    {
        ArgumentNullException.ThrowIfNull(value);

        var registeredValue = GetRegisteredValue(value.Name);
        _outputs.Add(registeredValue.Name);

        return registeredValue;
    }

    /// <summary>
    /// Removes a graph value from the model output list without deleting the value-info entry itself.
    /// </summary>
    /// <param name="name">Output name to unmark.</param>
    /// <returns><see langword="true"/> when the value was marked as an output.</returns>
    public bool RemoveOutput(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        return _outputs.Remove(name);
    }

    /// <summary>
    /// Removes a graph value from the model output list without deleting the value-info entry itself.
    /// </summary>
    /// <param name="value">Output value to unmark.</param>
    /// <returns><see langword="true"/> when the value was marked as an output.</returns>
    public bool RemoveOutput(OnnxValue value)
    {
        ArgumentNullException.ThrowIfNull(value);
        return RemoveOutput(value.Name);
    }

    /// <summary>
    /// Adds intermediate value-info for a tensor that should be described but not exposed as a model input or output.
    /// </summary>
    /// <typeparam name="T">Concrete value-type descriptor, typically <see cref="OnnxTensorType"/>.</typeparam>
    /// <param name="name">Intermediate wire name.</param>
    /// <param name="type">ONNX type descriptor for the intermediate value.</param>
    /// <returns>The intermediate value that can be connected to nodes.</returns>
    /// <exception cref="InvalidOperationException">Thrown when another intermediate value already uses <paramref name="name"/>.</exception>
    public OnnxValue AddValue<T>(string name, T type) where T : OnnxValueType
    {
        var intermediateValue = new OnnxValue<T>(name, type, null);
        return AddValue(intermediateValue);
    }

    /// <summary>
    /// Adds prepared value-info metadata and rewires matching loose-edge references to the typed value.
    /// </summary>
    /// <param name="value">Graph value to add.</param>
    /// <returns>The same value instance after registration.</returns>
    /// <exception cref="InvalidOperationException">Thrown when another value already uses <paramref name="value"/>'s name.</exception>
    public OnnxValue AddValue(OnnxValue value)
    {
        ArgumentNullException.ThrowIfNull(value);

        if (_values.Contains(value.Name))
        {
            throw new InvalidOperationException($"Value '{value.Name}' is already added into graph");
        }

        _values.Add(value);
        ReplaceNodeEdgeReferences(value.Name, value);
        _edges.Remove(value.Name);

        return value;
    }

    /// <summary>
    /// Replaces a graph value by name and updates input, output, and node wiring to the replacement value.
    /// </summary>
    /// <param name="name">Existing value name to replace.</param>
    /// <param name="value">Replacement graph value.</param>
    /// <returns>The replacement value.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the target value is missing or the replacement name collides.</exception>
    public OnnxValue ReplaceValue(string name, OnnxValue value)
    {
        ArgumentNullException.ThrowIfNull(name);
        ArgumentNullException.ThrowIfNull(value);

        if (!_values.Contains(name))
        {
            throw new InvalidOperationException($"No graph value found for '{name}'.");
        }

        if (_values.Contains(value.Name) && !StringComparer.Ordinal.Equals(name, value.Name))
        {
            throw new InvalidOperationException($"Value '{value.Name}' is already added into graph");
        }

        var wasInput = _inputs.Remove(name);
        var wasOutput = _outputs.Remove(name);

        _values.Replace(name, value);
        ReplaceNodeEdgeReferences(name, value);
        _edges.Remove(value.Name);

        if (wasInput)
        {
            _inputs.Add(value.Name);
        }

        if (wasOutput)
        {
            _outputs.Add(value.Name);
        }

        return value;
    }

    /// <summary>
    /// Removes value-info metadata and clears node input or output references to the same graph wire.
    /// </summary>
    /// <param name="name">Value name to remove.</param>
    /// <returns><see langword="true"/> when a value or node reference was removed.</returns>
    public bool RemoveValue(string name)
    {
        ArgumentNullException.ThrowIfNull(name);

        var removed = _values.Remove(name);
        _inputs.Remove(name);
        _outputs.Remove(name);
        removed |= _edges.Remove(name);
        removed |= RemoveNodeEdgeReferences(name);
        PruneUnusedLooseEdges();

        return removed;
    }

    /// <summary>
    /// Removes value-info metadata and clears node input or output references to the same graph wire.
    /// </summary>
    /// <param name="value">Value to remove.</param>
    /// <returns><see langword="true"/> when a value or node reference was removed.</returns>
    public bool RemoveValue(OnnxValue value)
    {
        ArgumentNullException.ThrowIfNull(value);
        return RemoveValue(value.Name);
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
    /// Removes a loose edge and clears node input or output references to the same graph wire.
    /// </summary>
    /// <param name="name">Edge name to remove.</param>
    /// <returns><see langword="true"/> when a loose edge or node reference was removed.</returns>
    public bool RemoveEdge(string name)
    {
        ArgumentNullException.ThrowIfNull(name);

        var removed = _edges.Remove(name);
        removed |= RemoveNodeEdgeReferences(name);
        PruneUnusedLooseEdges();

        return removed;
    }

    /// <summary>
    /// Removes a loose edge and clears node input or output references to the same graph wire.
    /// </summary>
    /// <param name="edge">Edge to remove.</param>
    /// <returns><see langword="true"/> when a loose edge or node reference was removed.</returns>
    public bool RemoveEdge(OnnxEdge edge)
    {
        ArgumentNullException.ThrowIfNull(edge);
        return RemoveEdge(edge.Name);
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
        ArgumentNullException.ThrowIfNull(node);

        if (_nodes.Contains(node.Name))
        {
            throw new InvalidOperationException($"Node '{node.Name}' is already added into graph");
        }

        _nodes.Add(node);
        return node;
    }

    /// <summary>
    /// Replaces a node by name while preserving its graph position.
    /// </summary>
    /// <param name="name">Existing node name to replace.</param>
    /// <param name="node">Replacement node.</param>
    /// <returns>The replacement node.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the target node is missing or the replacement name collides.</exception>
    public OnnxNode ReplaceNode(string name, OnnxNode node)
    {
        ArgumentNullException.ThrowIfNull(name);
        ArgumentNullException.ThrowIfNull(node);

        if (!_nodes.Replace(name, node))
        {
            throw new InvalidOperationException($"No graph node found for '{name}'.");
        }

        PruneUnusedLooseEdges();

        return node;
    }

    /// <summary>
    /// Removes a graph node and prunes loose edges that are no longer referenced by any remaining node.
    /// </summary>
    /// <param name="name">Node name to remove.</param>
    /// <returns><see langword="true"/> when a node was removed.</returns>
    public bool RemoveNode(string name)
    {
        ArgumentNullException.ThrowIfNull(name);

        var removed = _nodes.Remove(name);
        PruneUnusedLooseEdges();

        return removed;
    }

    /// <summary>
    /// Removes a graph node and prunes loose edges that are no longer referenced by any remaining node.
    /// </summary>
    /// <param name="node">Node to remove.</param>
    /// <returns><see langword="true"/> when a node was removed.</returns>
    public bool RemoveNode(OnnxNode node)
    {
        ArgumentNullException.ThrowIfNull(node);
        return RemoveNode(node.Name);
    }

    /// <summary>
    /// Reorders graph nodes, initializers, loose edges, and value-info entries into a deterministic topological order.
    /// </summary>
    /// <remarks>
    /// The method does not rename graph members and does not reorder the schema-defined inputs or outputs on individual nodes.
    /// Sorting is based on producer-consumer dependencies first, then on structural metadata such as operator type, attributes,
    /// tensor type, and shape. Existing graph order is used only as the final tie-breaker for structurally indistinguishable
    /// graph members.
    /// </remarks>
    public void SortTopologically()
    {
        var sortedNodes = GetTopologicallySortedNodes();
        Reorder(_nodes, sortedNodes);

        var edgeOrder = BuildEdgeOrder(sortedNodes);
        Reorder(_initializers, _initializers
            .Select((value, index) => new OrderedMember<OnnxTensor>(value, index))
            .OrderBy(x => GetOrder(edgeOrder, x.Value.Name))
            .ThenBy(x => GetTensorSignature(x.Value), StringComparer.Ordinal)
            .ThenBy(x => x.Index)
            .Select(x => x.Value));

        Reorder(_values, _values
            .Select((value, index) => new OrderedMember<OnnxValue>(value, index))
            .OrderBy(x => GetValueCategory(x.Value.Name))
            .ThenBy(x => GetOrder(edgeOrder, x.Value.Name))
            .ThenBy(x => GetValueSignature(x.Value), StringComparer.Ordinal)
            .ThenBy(x => x.Index)
            .Select(x => x.Value));

        Reorder(_edges, _edges
            .Select((value, index) => new OrderedMember<OnnxEdge>(value, index))
            .OrderBy(x => GetOrder(edgeOrder, x.Value.Name))
            .ThenBy(x => x.Index)
            .Select(x => x.Value));
    }

    private OnnxValue GetRegisteredValue(string name)
    {
        if (_values.TryGetValue(name, out var registeredValue))
        {
            return registeredValue;
        }

        throw new InvalidOperationException($"No graph value found for '{name}'.");
    }

    private bool ReplaceNodeEdgeReferences(string name, IOnnxGraphEdge value)
    {
        var replaced = false;

        foreach (var node in _nodes)
        {
            replaced |= node.ReplaceEdgeReference(name, value);
        }

        return replaced;
    }

    private bool RemoveNodeEdgeReferences(string name)
    {
        var removed = false;

        foreach (var node in _nodes)
        {
            removed |= node.RemoveEdgeReference(name);
        }

        return removed;
    }

    private void PruneUnusedLooseEdges()
    {
        var usedEdgeNames = _nodes
            .SelectMany(x => x.Inputs.Concat(x.Outputs))
            .Select(x => x.Name)
            .ToHashSet(StringComparer.Ordinal);

        foreach (var edge in _edges.ToArray())
        {
            if (!usedEdgeNames.Contains(edge.Name))
            {
                _edges.Remove(edge);
            }
        }
    }

    private IReadOnlyList<OnnxNode> GetTopologicallySortedNodes()
    {
        var originalIndexes = _nodes
            .Select((node, index) => (node, index))
            .ToDictionary(x => x.node, x => x.index);
        var producerByValue = new Dictionary<string, OnnxNode>(StringComparer.Ordinal);

        foreach (var node in _nodes)
        {
            foreach (var output in node.Outputs)
            {
                if (!producerByValue.TryAdd(output.Name, node))
                {
                    throw new InvalidOperationException($"Graph value '{output.Name}' is produced by more than one node.");
                }
            }
        }

        var depthByNode = new Dictionary<OnnxNode, int>();
        var visiting = new HashSet<OnnxNode>();
        foreach (var node in _nodes)
        {
            GetDepth(node);
        }

        return _nodes
            .OrderBy(node => depthByNode[node])
            .ThenBy(GetNodeStructuralSignature, StringComparer.Ordinal)
            .ThenBy(node => originalIndexes[node])
            .ToArray();

        int GetDepth(OnnxNode node)
        {
            if (depthByNode.TryGetValue(node, out var depth))
            {
                return depth;
            }

            if (!visiting.Add(node))
            {
                throw new InvalidOperationException("Graph contains a cycle and cannot be sorted topologically.");
            }

            depth = 0;
            foreach (var input in node.Inputs)
            {
                if (producerByValue.TryGetValue(input.Name, out var producer))
                {
                    depth = Math.Max(depth, GetDepth(producer) + 1);
                }
            }

            visiting.Remove(node);
            depthByNode[node] = depth;
            return depth;
        }
    }

    private Dictionary<string, int> BuildEdgeOrder(IReadOnlyList<OnnxNode> nodes)
    {
        var order = new Dictionary<string, int>(StringComparer.Ordinal);

        foreach (var node in nodes)
        {
            foreach (var input in node.Inputs)
            {
                Add(input.Name);
            }

            foreach (var output in node.Outputs)
            {
                Add(output.Name);
            }
        }

        return order;

        void Add(string name)
        {
            if (!order.ContainsKey(name))
            {
                order[name] = order.Count;
            }
        }
    }

    private int GetValueCategory(string name)
    {
        if (_inputs.Contains(name))
        {
            return 0;
        }

        if (_outputs.Contains(name))
        {
            return 2;
        }

        return 1;
    }

    private static int GetOrder(IReadOnlyDictionary<string, int> order, string name)
    {
        return order.TryGetValue(name, out var value) ? value : int.MaxValue;
    }

    private static string GetNodeStructuralSignature(OnnxNode node)
    {
        var attributes = string.Join(
            ";",
            node.Attributes
                .Select(GetAttributeSignature)
                .OrderBy(static x => x, StringComparer.Ordinal));

        return string.Join(
            "|",
            node.Domain,
            node.OpType,
            node.Inputs.Count.ToString(System.Globalization.CultureInfo.InvariantCulture),
            node.Outputs.Count.ToString(System.Globalization.CultureInfo.InvariantCulture),
            attributes);
    }

    private static string GetAttributeSignature(OnnxAttribute attribute)
    {
        var value = attribute.GetValue();
        return $"{attribute.Name}:{value?.GetType().FullName}:{FormatAttributeValue(value)}";
    }

    private static string FormatAttributeValue(object? value)
    {
        if (value is null)
        {
            return string.Empty;
        }

        if (value is System.Collections.IEnumerable enumerable && value is not string)
        {
            return string.Join(",", enumerable.Cast<object>().Select(FormatAttributeValue));
        }

        return Convert.ToString(value, System.Globalization.CultureInfo.InvariantCulture) ?? string.Empty;
    }

    private static string GetTensorSignature(OnnxTensor tensor)
    {
        return $"{tensor.DataType.FullName}[{string.Join(",", tensor.Shape)}]";
    }

    private static string GetValueSignature(OnnxValue value)
    {
        return $"{value.GetType().FullName}:{value.Type}";
    }

    private static void Reorder<T>(LazyDictionary<string, T> collection, IEnumerable<T> values) where T : IOnnxGraphEdge
    {
        var sorted = values.ToArray();
        collection.Clear();

        foreach (var value in sorted)
        {
            collection.Add(value);
        }
    }

    private static void Reorder(LazyDictionary<string, OnnxNode> collection, IEnumerable<OnnxNode> values)
    {
        var sorted = values.ToArray();
        collection.Clear();

        foreach (var value in sorted)
        {
            collection.Add(value);
        }
    }

    internal GraphProto ToProto()
    {
        var newGraph = _graph.Clone();
        newGraph.Name = Name;

        newGraph.Initializer.Set(_initializers.Select(x => x.ToProto()));
        newGraph.ValueInfo.Set(IntermediateValues.Select(x => x.ToProto()));
        newGraph.Input.Set(Inputs.Select(x => x.ToProto()));
        newGraph.Output.Set(Outputs.Select(x => x.ToProto()));
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
                Values={FormatCollection(IntermediateValues).Indent(1)},
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

    private readonly record struct OrderedMember<T>(T Value, int Index);
}

/// <summary>
/// Entry point for generated wrappers in the ONNX ML operator domain.
/// 
/// <para>ai.onnx.ml</para>
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
/// 
/// <para>com.microsoft.*</para>
/// </summary>
public readonly struct MicrosoftDomain(OnnxGraph graph)
{
    /// <summary>
    /// Gets the graph that generated Microsoft-domain wrapper calls will mutate.
    /// </summary>
    public readonly OnnxGraph Graph = graph;

    /// <summary>
    /// com.ms.internal.nhwc
    /// </summary>
    public MicrosoftInternalDomain Internal => new(Graph);

    /// <summary>
    /// com.microsoft.nhwc
    /// </summary>
    public MicrosoftInternalNHWCDomain NHWC => new(Graph);

    /// <summary>
    /// com.microsoft.nchwc
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
    /// com.ms.internal.nhwc
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
