using System.Collections.ObjectModel;
using Onnx;
using Onnxify.Data;
using Onnxify.Helpers;

namespace Onnxify;

/// <summary>
/// Represents one ONNX operator invocation and its ordered input, output, and attribute lists.
/// </summary>
/// <remarks>
/// Generic nodes are useful for operators without a generated wrapper or for preserving unknown operators from loaded models. Generated operator classes derive from this type and use the protected helpers to keep schema-ordered inputs, outputs, and attributes synchronized.
/// </remarks>
public class OnnxNode : IOnnxGraphNode
{
    /// <summary>
    /// Gets the graph-local node name used for diagnostics and duplicate detection.
    /// </summary>
    public string Name { get; init; }

    /// <summary>
    /// Gets or sets the ONNX operator type resolved within <see cref="Domain"/>.
    /// </summary>
    public string OpType { get; set; }

    /// <summary>
    /// Gets or sets the operator domain. The empty string selects the standard ONNX domain.
    /// </summary>
    public string Domain { get; set; }

    /// <summary>
    /// Gets or sets the optional ONNX node documentation string.
    /// </summary>
    public string DocString { get; set; }

    /// <summary>
    /// Gets node inputs in ONNX schema order.
    /// </summary>
    public IReadOnlyList<IOnnxGraphEdge> Inputs => _inputs;

    /// <summary>
    /// Gets node outputs in ONNX schema order.
    /// </summary>
    public IReadOnlyList<IOnnxGraphEdge> Outputs => _outputs;

    /// <summary>
    /// Gets node attributes keyed by ONNX attribute name.
    /// </summary>
    public IReadOnlyList<OnnxAttribute> Attributes => _attributes;

    private readonly LazyDictionary<string, IOnnxGraphEdge> _inputs = new(x => x.Name, EqualityComparer<string>.Default);
    private readonly LazyDictionary<string, IOnnxGraphEdge> _outputs = new(x => x.Name, EqualityComparer<string>.Default);
    private readonly LazyDictionary<string, OnnxAttribute> _attributes = new(x => x.Name, EqualityComparer<string>.Default);

    private readonly NodeProto _node;

    /// <summary>
    /// Creates a generic node for an ONNX operator.
    /// </summary>
    /// <param name="name">Graph-local node name.</param>
    /// <param name="opType">Operator type, such as <c>Conv</c> or <c>MatMul</c>.</param>
    /// <param name="domain">Operator domain. Use an empty string for standard ONNX operators.</param>
    /// <param name="docString">Optional node documentation string stored in ONNX.</param>
    /// <param name="inputs">Input wires in operator schema order.</param>
    /// <param name="outputs">Output wires in operator schema order.</param>
    /// <param name="attributes">Operator attributes keyed by ONNX attribute name.</param>
    public OnnxNode(
        string name,
        string opType,
        string domain,
        string docString,
        IEnumerable<IOnnxGraphEdge> inputs,
        IEnumerable<IOnnxGraphEdge> outputs,
        IEnumerable<OnnxAttribute> attributes
    ) : this(
        name: name,
        opType: opType,
        domain: domain,
        docString: docString,
        inputs: inputs,
        outputs: outputs,
        attributes: attributes,
        proto: null
    )
    { }

    internal OnnxNode(
        string name,
        string opType,
        string domain,
        string docString,
        IEnumerable<IOnnxGraphEdge> inputs,
        IEnumerable<IOnnxGraphEdge> outputs,
        IEnumerable<OnnxAttribute> attributes,
        NodeProto? proto
    )
    {
        _node = proto ?? new NodeProto
        {
            Name = name,
        };

        Name = name;
        OpType = opType;
        Domain = domain;
        DocString = docString;

        foreach (var x in inputs)
        {
            _inputs.Add(x);
        }

        foreach (var x in outputs)
        {
            _outputs.Add(x);
        }

        foreach (var x in attributes)
        {
            _attributes.Add(x);
        }
    }

    internal NodeProto ToProto()
    {
        var newNode = _node.Clone();
        newNode.Name = Name;
        newNode.OpType = OpType;
        newNode.Domain = Domain;
        newNode.DocString = DocString;

        newNode.Input.Set(Inputs.Select(x => x.Name));
        newNode.Output.Set(Outputs.Select(x => x.Name));
        newNode.Attribute.Set(Attributes.Select(x => x.ToProto()));

        return newNode;
    }

    internal static OnnxNode FromProto(NodeProto node, OnnxGraph graph)
    {
        var typedNode = OnnxNodeHelper.TryFromProto(node, graph);

        if (typedNode is not null)
        {
            return typedNode;
        }

        var options = graph.GetOptions();

        var name = node.Name;
        var opType = node.OpType;
        var domain = node.Domain;
        var docString = node.DocString;
        var inputs = new List<IOnnxGraphEdge>();
        var outputs = new List<IOnnxGraphEdge>();
        var attributes = new List<OnnxAttribute>();

        foreach (var x in node.Input)
        {
            var value = graph.GetValue(x) ?? throw new InvalidOperationException($"No graph value found for '{x}'.");
            inputs.Add(value);
        }

        foreach (var x in node.Output)
        {
            var value = graph.GetValue(x) ?? throw new InvalidOperationException($"No graph value found for '{x}'.");
            outputs.Add(value);
        }

        foreach (var attribute in node.Attribute)
        {
            var value = OnnxHelper.FromProto(attribute, options);
            attributes.Add(value);
        }

        return new OnnxNode(
            name: name,
            opType: opType,
            domain: domain,
            docString: docString,
            inputs: inputs,
            outputs: outputs,
            attributes: attributes,
            proto: node
        );
    }

    /// <summary>
    /// Replaces the required input at a schema index.
    /// </summary>
    /// <param name="index">Zero-based schema input position.</param>
    /// <param name="value">Wire to write at that position.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="index"/> is negative.</exception>
    protected void SetInput(int index, IOnnxGraphEdge value)
    {
        ArgumentNullException.ThrowIfNull(value);
        ArgumentOutOfRangeException.ThrowIfNegative(index);

        if (index < _inputs.Count)
        {
            var existing = _inputs[index];
            _inputs.Remove(existing);
        }

        _inputs.Add(value);
    }

    /// <summary>
    /// Replaces or removes an optional input at a schema index.
    /// </summary>
    /// <param name="index">Zero-based schema input position.</param>
    /// <param name="value">Wire to write, or <see langword="null"/> to omit the optional input.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="index"/> is negative.</exception>
    protected void SetOptionalInput(int index, IOnnxGraphEdge? value)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);

        if (index < _inputs.Count)
        {
            var existing = _inputs[index];
            _inputs.Remove(existing);
        }

        if (value != null)
        {
            _inputs.Add(value);
        }
    }

    /// <summary>
    /// Replaces the required output at a schema index.
    /// </summary>
    /// <param name="index">Zero-based schema output position.</param>
    /// <param name="value">Wire to write at that position.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="index"/> is negative.</exception>
    protected void SetOutput(int index, IOnnxGraphEdge value)
    {
        ArgumentNullException.ThrowIfNull(value);
        ArgumentOutOfRangeException.ThrowIfNegative(index);

        if (index < _outputs.Count)
        {
            var existing = _outputs[index];
            _outputs.Remove(existing);
        }

        _outputs.Add(value);
    }

    /// <summary>
    /// Replaces or removes an optional output at a schema index.
    /// </summary>
    /// <param name="index">Zero-based schema output position.</param>
    /// <param name="value">Wire to write, or <see langword="null"/> to omit the optional output.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="index"/> is negative.</exception>
    protected void SetOptionalOutput(int index, IOnnxGraphEdge? value)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);

        if (index < _outputs.Count)
        {
            var existing = _outputs[index];
            _outputs.Remove(existing);
        }

        if (value != null)
        {
            _outputs.Add(value);
        }
    }

    /// <summary>
    /// Reads all inputs belonging to a variadic schema parameter.
    /// </summary>
    /// <param name="startIndex">First index of the variadic input segment.</param>
    /// <returns>A read-only snapshot of inputs from <paramref name="startIndex"/> onward.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="startIndex"/> is negative.</exception>
    protected IReadOnlyList<IOnnxGraphEdge> GetVariadicInputs(int startIndex)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(startIndex);

        if (startIndex >= _inputs.Count)
        {
            return Array.Empty<IOnnxGraphEdge>();
        }

        return new ReadOnlyCollection<IOnnxGraphEdge>(_inputs.Skip(startIndex).ToArray());
    }

    /// <summary>
    /// Replaces the variadic input segment starting at a schema index.
    /// </summary>
    /// <param name="startIndex">First index of the variadic input segment.</param>
    /// <param name="values">Input wires to store from <paramref name="startIndex"/> onward.</param>
    /// <exception cref="ArgumentException">Thrown when <paramref name="values"/> contains <see langword="null"/>.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="startIndex"/> is negative.</exception>
    protected void SetVariadicInputs(int startIndex, IEnumerable<IOnnxGraphEdge> values)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(startIndex);
        ArgumentNullException.ThrowIfNull(values);

        var newValues = values.ToArray();

        if (newValues.Any(x => x is null))
        {
            throw new ArgumentException("Variadic inputs cannot contain null values.", nameof(values));
        }

        while (_inputs.Count > startIndex)
        {
            _inputs.Remove(_inputs[startIndex]);
        }

        foreach (var value in newValues)
        {
            _inputs.Add(value);
        }
    }

    /// <summary>
    /// Reads all outputs belonging to a variadic schema parameter.
    /// </summary>
    /// <param name="startIndex">First index of the variadic output segment.</param>
    /// <returns>A read-only snapshot of outputs from <paramref name="startIndex"/> onward.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="startIndex"/> is negative.</exception>
    protected IReadOnlyList<IOnnxGraphEdge> GetVariadicOutputs(int startIndex)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(startIndex);

        if (startIndex >= _outputs.Count)
        {
            return Array.Empty<IOnnxGraphEdge>();
        }

        return new ReadOnlyCollection<IOnnxGraphEdge>(_outputs.Skip(startIndex).ToArray());
    }

    /// <summary>
    /// Replaces the variadic output segment starting at a schema index.
    /// </summary>
    /// <param name="startIndex">First index of the variadic output segment.</param>
    /// <param name="values">Output wires to store from <paramref name="startIndex"/> onward.</param>
    /// <exception cref="ArgumentException">Thrown when <paramref name="values"/> contains <see langword="null"/>.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="startIndex"/> is negative.</exception>
    protected void SetVariadicOutputs(int startIndex, IEnumerable<IOnnxGraphEdge> values)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(startIndex);
        ArgumentNullException.ThrowIfNull(values);

        var newValues = values.ToArray();

        if (newValues.Any(x => x is null))
        {
            throw new ArgumentException("Variadic outputs cannot contain null values.", nameof(values));
        }

        while (_outputs.Count > startIndex)
        {
            _outputs.Remove(_outputs[startIndex]);
        }

        foreach (var value in newValues)
        {
            _outputs.Add(value);
        }
    }

    /// <summary>
    /// Checks whether an attribute is present without applying an ONNX schema default.
    /// </summary>
    /// <param name="name">Attribute name exactly as it appears in the ONNX schema.</param>
    /// <returns><see langword="true"/> when the node carries the attribute.</returns>
    protected bool HasAttribute(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        return _attributes.Contains(name);
    }

    /// <summary>
    /// Reads an attribute value and converts compatible numeric values to the requested CLR type.
    /// </summary>
    /// <typeparam name="T">Expected CLR value type.</typeparam>
    /// <param name="name">Attribute name exactly as it appears in the ONNX schema.</param>
    /// <returns>The stored attribute value.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the attribute is absent.</exception>
    protected T GetAttribute<T>(string name) where T : notnull
    {
        ArgumentNullException.ThrowIfNull(name);

        if (_attributes.TryGetValue(name, out var attr))
        {
            var value = attr.GetValue();

            if (value is T t)
            {
                return t;
            }

            return (T)Convert.ChangeType(value, typeof(T));
        }

        throw new InvalidOperationException($"No attribute with name '{name}'");
    }

    /// <summary>
    /// Adds or replaces an attribute using ONNX attribute type inference from <typeparamref name="T"/>.
    /// </summary>
    /// <typeparam name="T">CLR value type supported by ONNX attribute serialization.</typeparam>
    /// <param name="name">Attribute name exactly as it appears in the ONNX schema.</param>
    /// <param name="value">Attribute value to serialize.</param>
    protected void SetAttribute<T>(string name, T value) where T : notnull
    {
        ArgumentNullException.ThrowIfNull(name);
        _attributes[name] = new OnnxAttribute<T>(name, (T)value);
    }

    /// <summary>
    /// Removes an explicitly stored attribute so the operator schema default, if any, can apply downstream.
    /// </summary>
    /// <param name="name">Attribute name exactly as it appears in the ONNX schema.</param>
    /// <returns><see langword="true"/> when an attribute was removed.</returns>
    protected bool RemoveAttribute(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        return _attributes.Remove(name);
    }

    /// <summary>
    /// Returns a diagnostic view of the node, including ordered inputs, outputs, and attributes.
    /// </summary>
    /// <returns>A multiline string intended for inspection, not for stable serialization.</returns>
    public override string ToString()
    {
        var domain = string.IsNullOrWhiteSpace(Domain) ? "<default>" : Domain;

        return $"""
            OnnxNode(
                Name={Name},
                OpType={OpType},
                Domain={domain},
                Inputs=[
                    {string.Join(",\n", Inputs).Indent(2)}
                ],
                Outputs=[
                    {string.Join(",\n", Outputs).Indent(2)}
                ],
                Attributes=[
                    {string.Join(",\n", Attributes).Indent(2)}
                ]
                Doc={(!string.IsNullOrWhiteSpace(DocString) ? DocString : "<missing>")}
            )
            """;
    }
}

