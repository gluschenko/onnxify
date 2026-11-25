using System.Reflection;
using System.Reflection.Emit;
using Onnxify.Compiler;

namespace Onnxify.TorchSharp;

/// <summary>
/// Creates compiler-neutral C# TorchSharp sources from runtime modules. Decompilation and syntax
/// scanning remain in <see cref="Onnxify.Compiler.Compiler"/>.
/// </summary>
public static class TorchSharpCompilerAdapter
{
    private static readonly IReadOnlyDictionary<short, OpCode> _opCodes = typeof(OpCodes)
        .GetFields(BindingFlags.Public | BindingFlags.Static)
        .Where(static field => field.FieldType == typeof(OpCode))
        .Select(static field => (OpCode)field.GetValue(null)!)
        .ToDictionary(static opCode => opCode.Value);

    /// <summary>Creates a compiler source for the module's <c>forward</c> method.</summary>
    public static CSharpTorchSharpSource CreateSource(
        global::TorchSharp.torch.nn.Module module,
        string methodName = "forward",
        string? document = null
    )
    {
        ArgumentNullException.ThrowIfNull(module);
        ArgumentException.ThrowIfNullOrWhiteSpace(methodName);

        var descriptor = CreateDescriptor(
            module: module,
            methodName: methodName,
            document: document,
            activeModules: new HashSet<global::TorchSharp.torch.nn.Module>(ReferenceEqualityComparer.Instance));
        return new CSharpTorchSharpSource(descriptor);
    }

    private static CompilerTorchSharpModuleDescriptor CreateDescriptor(
        global::TorchSharp.torch.nn.Module module,
        string methodName,
        string? document,
        HashSet<global::TorchSharp.torch.nn.Module> activeModules
    )
    {
        if (!activeModules.Add(module))
        {
            throw new InvalidOperationException(
                $"The TorchSharp module hierarchy contains a cycle at '{module.GetType().FullName}'.");
        }

        try
        {
            var moduleType = module.GetType();
            var method = moduleType
                .GetMethods(BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)
                .Where(x => string.Equals(x.Name, methodName, StringComparison.Ordinal))
                .OrderBy(x => x.MetadataToken)
                .FirstOrDefault()
                ?? throw new InvalidOperationException(
                    $"Method '{methodName}' was not found on '{moduleType.FullName}'.");

            var assemblyPath = moduleType.Assembly.Location;
            if (string.IsNullOrWhiteSpace(assemblyPath))
            {
                throw new InvalidOperationException(
                    $"Module type '{moduleType.FullName}' does not have a loadable assembly location.");
            }

            var inputs = GetInputs(moduleType, method);
            var outputs = GetOutputs(moduleType, method);
            var stateMembers = GetStateMembers(module, includeRegisteredTensorState: activeModules.Count == 1);
            var childModules = GetChildModules(module, activeModules);
            var helpers = GetReachableHelperMethods(moduleType, method)
                .Select(static helper => new CompilerTorchSharpHelperMethodDescriptor(helper.Name))
                .ToArray();

            var descriptor = new CompilerTorchSharpModuleDescriptor(
                assemblyPath,
                moduleType.FullName ?? moduleType.Name,
                methodName,
                method.MetadataToken,
                document ?? assemblyPath,
                inputs,
                outputs,
                stateMembers,
                childModules,
                helpers);

            return descriptor;
        }
        finally
        {
            activeModules.Remove(module);
        }
    }

    private static IReadOnlyList<MethodInfo> GetReachableHelperMethods(Type moduleType, MethodInfo rootMethod)
    {
        var candidates = moduleType
            .GetMethods(BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)
            .Where(method => method.DeclaringType == moduleType && !method.IsSpecialName)
            .GroupBy(static method => method.Name, StringComparer.Ordinal)
            .ToDictionary(group => group.Key, group => group.OrderBy(static method => method.MetadataToken).First(), StringComparer.Ordinal);
        var reachable = new Dictionary<string, MethodInfo>(StringComparer.Ordinal);
        var pending = new Queue<MethodInfo>(GetCalledMethods(rootMethod)
            .Where(method => method.DeclaringType == moduleType));

        while (pending.Count > 0)
        {
            var called = pending.Dequeue();
            if (called.IsSpecialName
                || string.Equals(called.Name, rootMethod.Name, StringComparison.Ordinal)
                || !candidates.TryGetValue(called.Name, out var helper)
                || !reachable.TryAdd(helper.Name, helper))
            {
                continue;
            }

            foreach (var nested in GetCalledMethods(helper).Where(method => method.DeclaringType == moduleType))
                pending.Enqueue(nested);
        }

        return reachable.Values.OrderBy(static helper => helper.MetadataToken).ToArray();
    }

    private static IReadOnlyList<MethodInfo> GetCalledMethods(MethodInfo method)
    {
        var bytes = method.GetMethodBody()?.GetILAsByteArray();
        if (bytes is null || bytes.Length == 0)
            return [];

        var calledMethods = new List<MethodInfo>();
        var offset = 0;
        while (offset < bytes.Length)
        {
            short code = bytes[offset++];
            if (code == 0xFE)
                code = unchecked((short)(0xFE00 | bytes[offset++]));
            if (!_opCodes.TryGetValue(code, out var opCode))
                break;

            if (opCode.OperandType == OperandType.InlineMethod)
            {
                var metadataToken = BitConverter.ToInt32(bytes, offset);
                try
                {
                    if (method.Module.ResolveMethod(
                        metadataToken,
                        method.DeclaringType?.GetGenericArguments(),
                        method.IsGenericMethod ? method.GetGenericArguments() : null) is MethodInfo called)
                    {
                        calledMethods.Add(called);
                    }
                }
                catch (ArgumentException)
                {
                    // Ignore method tokens that cannot be resolved in the current generic context.
                }
            }

            offset += GetOperandSize(opCode.OperandType, bytes, offset);
        }

        return calledMethods;
    }

    private static int GetOperandSize(OperandType operandType, byte[] bytes, int offset) => operandType switch
    {
        OperandType.InlineNone => 0,
        OperandType.ShortInlineBrTarget or OperandType.ShortInlineI or OperandType.ShortInlineVar => 1,
        OperandType.InlineVar => 2,
        OperandType.InlineI or OperandType.InlineBrTarget or OperandType.InlineField
            or OperandType.InlineMethod or OperandType.InlineSig or OperandType.InlineString
            or OperandType.InlineTok or OperandType.InlineType or OperandType.ShortInlineR => 4,
        OperandType.InlineI8 or OperandType.InlineR => 8,
        OperandType.InlineSwitch => checked(4 + (BitConverter.ToInt32(bytes, offset) * 4)),
        _ => throw new InvalidOperationException($"Unsupported IL operand type '{operandType}'."),
    };

    private static IReadOnlyList<CompilerTorchSharpValueDescriptor> GetInputs(
        Type moduleType,
        MethodInfo method
    )
    {
        var attributes = moduleType
            .GetCustomAttributes<ModuleInputAttribute>(inherit: true)
            .OrderBy(x => x.Name, StringComparer.Ordinal)
            .ToArray();
        return method.GetParameters()
            .Select(
                (parameter, index) =>
                {
                    var attribute = index < attributes.Length ? attributes[index] : null;
                    return new CompilerTorchSharpValueDescriptor(
                        parameter.Name ?? $"input{index}",
                        attribute is null
                            ? TypeFromCSharpType(parameter.ParameterType)
                            : TensorType(attribute.DataType, attribute.Dimensions),
                        CSharpTypeName(parameter.ParameterType));
                })
            .ToArray();
    }

    private static IReadOnlyList<CompilerTorchSharpValueDescriptor> GetOutputs(
        Type moduleType,
        MethodInfo method
    )
    {
        var attributes = moduleType
            .GetCustomAttributes<ModuleOutputAttribute>(inherit: true)
            .OrderBy(x => x.Name, StringComparer.Ordinal)
            .ToArray();
        if (attributes.Length != 0)
        {
            return attributes
                .Select(
                    x => new CompilerTorchSharpValueDescriptor(
                        x.Name,
                        TensorType(x.DataType, x.Dimensions),
                        CSharpTypeName(method.ReturnType)))
                .ToArray();
        }

        return [
            new CompilerTorchSharpValueDescriptor(
                "output",
                TypeFromCSharpType(method.ReturnType),
                CSharpTypeName(method.ReturnType)),
        ];
    }

    private static CompilerType TypeFromCSharpType(Type type)
    {
        var nullableType = Nullable.GetUnderlyingType(type);
        if (nullableType is not null)
        {
            return new CompilerNullableType(TypeFromCSharpType(nullableType));
        }

        if (type == typeof(global::TorchSharp.torch.Tensor)
            || typeof(global::TorchSharp.torch.Tensor).IsAssignableFrom(type))
        {
            return TensorType(null, null);
        }

        if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(ValueTuple<,>))
        {
            return new CompilerTupleType(type.GetGenericArguments().Select(TypeFromCSharpType));
        }

        if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(ValueTuple<,,>))
        {
            return new CompilerTupleType(type.GetGenericArguments().Select(TypeFromCSharpType));
        }

        if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(ValueTuple<,,,>))
        {
            return new CompilerTupleType(type.GetGenericArguments().Select(TypeFromCSharpType));
        }

        if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(ValueTuple<,,,,>))
        {
            return new CompilerTupleType(type.GetGenericArguments().Select(TypeFromCSharpType));
        }

        if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(ValueTuple<,,,,,>))
        {
            return new CompilerTupleType(type.GetGenericArguments().Select(TypeFromCSharpType));
        }

        if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(ValueTuple<,,,,,,>))
        {
            return new CompilerTupleType(type.GetGenericArguments().Select(TypeFromCSharpType));
        }

        return type == typeof(bool) ? new CompilerScalarType(CompilerElementType.Boolean)
            : type == typeof(int) ? new CompilerScalarType(CompilerElementType.Int32)
            : type == typeof(long) ? new CompilerScalarType(CompilerElementType.Int64)
            : type == typeof(float) ? new CompilerScalarType(CompilerElementType.Float32)
            : type == typeof(double) ? new CompilerScalarType(CompilerElementType.Float64)
            : new CompilerOpaqueType("CSharp", type.FullName ?? type.Name);
    }

    private static string CSharpTypeName(Type type)
    {
        var nullableType = Nullable.GetUnderlyingType(type);
        if (nullableType is not null)
        {
            return $"{CSharpTypeName(nullableType)}?";
        }

        if (type == typeof(global::TorchSharp.torch.Tensor)) return "global::TorchSharp.torch.Tensor";
        if (type == typeof(bool)) return "bool";
        if (type == typeof(int)) return "int";
        if (type == typeof(long)) return "long";
        if (type == typeof(float)) return "float";
        if (type == typeof(double)) return "double";
        if (type == typeof(string)) return "string";

        if (type.IsGenericType && type.GetGenericTypeDefinition().FullName?.StartsWith("System.ValueTuple`", StringComparison.Ordinal) == true)
        {
            var tupleTypes = type.GetGenericArguments().Select(CSharpTypeName);
            return $"({string.Join(", ", tupleTypes)})";
        }

        return $"global::{type.FullName?.Replace('+', '.') ?? type.Name}";
    }

    private static IReadOnlyList<CompilerTorchSharpStateMemberDescriptor> GetStateMembers(
        global::TorchSharp.torch.nn.Module module,
        bool includeRegisteredTensorState
    )
    {
        var members = new List<CompilerTorchSharpStateMemberDescriptor>();
        foreach (var field in module.GetType()
            .GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
            .Where(field => field.DeclaringType == module.GetType()))
        {
            var value = field.GetValue(module);
            if (value is global::TorchSharp.torch.Tensor)
            {
                // Tensor-valued module fields are represented by state_dict() below, which
                // supplies stable hierarchical names and includes registered child parameters.
                continue;
            }

            if (TryCreateScalarLiteral(value, out var literal, out var type))
            {
                members.Add(
                    new CompilerTorchSharpStateMemberDescriptor(
                        field.Name,
                        CompilerStateMemberKind.Initializer,
                        type,
                        literal,
                        field.FieldType.FullName));
            }
        }

        foreach (var (name, tensor) in module.state_dict())
        {
            if (tensor is null || tensor.IsInvalid)
            {
                continue;
            }

            members.Add(
                new CompilerTorchSharpStateMemberDescriptor(
                    name,
                    includeRegisteredTensorState ? CompilerStateMemberKind.Initializer : CompilerStateMemberKind.Buffer,
                    TensorType(tensor.dtype, tensor.GetShape().Select(x => new TensorDimension(x))),
                    includeRegisteredTensorState ? CreateTensorLiteral(tensor) : null,
                    "global::TorchSharp.torch.Tensor"));
        }

        if (includeRegisteredTensorState)
        {
            AddLstmOnnxInitializers(module, string.Empty, members);
        }

        return members
            .GroupBy(x => x.Name, StringComparer.Ordinal)
            .Select(x => x.First())
            .ToArray();
    }

    private static CompilerTensorLiteral CreateTensorLiteral(global::TorchSharp.torch.Tensor tensor)
    {
        using var cpuTensor = tensor.detach().cpu().contiguous();
        var elementType = ToCompilerElementType(cpuTensor.dtype.ToString());
        CompilerScalarLiteral[] values = cpuTensor.dtype switch
        {
            global::TorchSharp.torch.ScalarType.Bool => cpuTensor.data<bool>().ToArray()
                .Select(value => (CompilerScalarLiteral)new CompilerBooleanLiteral(value)).ToArray(),
            global::TorchSharp.torch.ScalarType.Byte => cpuTensor.data<byte>().ToArray()
                .Select(value => (CompilerScalarLiteral)new CompilerUnsignedIntegerLiteral(elementType, value)).ToArray(),
            global::TorchSharp.torch.ScalarType.Int8 => cpuTensor.data<sbyte>().ToArray()
                .Select(value => (CompilerScalarLiteral)new CompilerSignedIntegerLiteral(elementType, value)).ToArray(),
            global::TorchSharp.torch.ScalarType.Int16 => cpuTensor.data<short>().ToArray()
                .Select(value => (CompilerScalarLiteral)new CompilerSignedIntegerLiteral(elementType, value)).ToArray(),
            global::TorchSharp.torch.ScalarType.Int32 => cpuTensor.data<int>().ToArray()
                .Select(value => (CompilerScalarLiteral)new CompilerSignedIntegerLiteral(elementType, value)).ToArray(),
            global::TorchSharp.torch.ScalarType.Int64 => cpuTensor.data<long>().ToArray()
                .Select(value => (CompilerScalarLiteral)new CompilerSignedIntegerLiteral(elementType, value)).ToArray(),
            global::TorchSharp.torch.ScalarType.Float32 => cpuTensor.data<float>().ToArray()
                .Select(value => (CompilerScalarLiteral)new CompilerFloatingPointLiteral(elementType, value)).ToArray(),
            global::TorchSharp.torch.ScalarType.Float64 => cpuTensor.data<double>().ToArray()
                .Select(value => (CompilerScalarLiteral)new CompilerFloatingPointLiteral(elementType, value)).ToArray(),
            _ => throw new NotSupportedException($"TorchSharp tensor dtype '{cpuTensor.dtype}' cannot be captured as a compiler initializer."),
        };

        return new CompilerTensorLiteral(
            elementType,
            cpuTensor.GetShape().Select(dimension => (CompilerDimension)new CompilerFixedDimension(dimension)),
            values);
    }

    private static void AddLstmOnnxInitializers(
        global::TorchSharp.torch.nn.Module module,
        string prefix,
        List<CompilerTorchSharpStateMemberDescriptor> members)
    {
        if (module.GetType().FullName?.EndsWith(".LSTM", StringComparison.Ordinal) == true)
        {
            CreateLstmOnnxInitializers(module, prefix, members);
            return;
        }

        if (module.GetType().FullName?.EndsWith(".Linear", StringComparison.Ordinal) == true)
        {
            CreateLinearOnnxInitializers(module, prefix, members);
            return;
        }

        foreach (var field in module.GetType()
            .GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
            .Where(field => field.DeclaringType == module.GetType())
            .Where(field => typeof(global::TorchSharp.torch.nn.Module).IsAssignableFrom(field.FieldType)))
        {
            if (field.GetValue(module) is global::TorchSharp.torch.nn.Module child)
            {
                AddLstmOnnxInitializers(child, $"{prefix}{field.Name}.", members);
            }
        }
    }

    private static void CreateLinearOnnxInitializers(
        global::TorchSharp.torch.nn.Module module,
        string prefix,
        List<CompilerTorchSharpStateMemberDescriptor> members)
    {
        var weight = module.state_dict().FirstOrDefault(pair => string.Equals(pair.Key, "weight", StringComparison.Ordinal)).Value;
        if (weight is null || weight.IsInvalid || weight.dtype != global::TorchSharp.torch.ScalarType.Float32)
        {
            throw new NotSupportedException($"TorchSharp Linear module '{prefix}' requires a registered Float32 weight tensor.");
        }

        using var cpuWeight = weight.detach().cpu().contiguous();
        var shape = cpuWeight.GetShape();
        if (shape.Length != 2)
        {
            throw new NotSupportedException($"TorchSharp Linear module '{prefix}' requires a rank-2 weight tensor.");
        }

        var outputSize = checked((int)shape[0]);
        var inputSize = checked((int)shape[1]);
        var source = cpuWeight.data<float>().ToArray();
        var transposed = new CompilerScalarLiteral[source.Length];
        var offset = 0;
        for (var column = 0; column < inputSize; column++)
        {
            for (var row = 0; row < outputSize; row++)
            {
                transposed[offset++] = new CompilerFloatingPointLiteral(
                    CompilerElementType.Float32,
                    source[row * inputSize + column]);
            }
        }

        var tensor = new CompilerTensorLiteral(
            CompilerElementType.Float32,
            [new CompilerFixedDimension(inputSize), new CompilerFixedDimension(outputSize)],
            transposed);
        members.Add(new CompilerTorchSharpStateMemberDescriptor(
            prefix + "__oxy_W",
            CompilerStateMemberKind.Initializer,
            new CompilerTensorType(CompilerElementType.Float32, tensor.Dimensions),
            tensor,
            "global::TorchSharp.torch.Tensor"));
    }

    private static void CreateLstmOnnxInitializers(
        global::TorchSharp.torch.nn.Module module,
        string prefix,
        List<CompilerTorchSharpStateMemberDescriptor> members)
    {
        var state = module.state_dict().ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal);
        var hiddenSize = ReadIntField(module, "_hidden_size");
        var layers = ReadIntField(module, "_num_layers");
        var bidirectional = ReadBoolField(module, "_bidirectional");
        if (layers != 1)
        {
            throw new NotSupportedException("TorchSharp LSTM reverse lowering currently supports one recurrent layer per module.");
        }

        var directions = bidirectional ? 2 : 1;
        var weightInput = RequireFloatTensor(state, "weight_ih_l0", prefix);
        var inputSize = weightInput.Dimensions[1] is CompilerFixedDimension inputDimension
            ? checked((int)inputDimension.Value)
            : throw new NotSupportedException("TorchSharp LSTM input weights must have a fixed input dimension.");
        var wValues = new List<CompilerScalarLiteral>();
        var rValues = new List<CompilerScalarLiteral>();
        var bValues = new List<CompilerScalarLiteral>();
        for (var direction = 0; direction < directions; direction++)
        {
            var suffix = direction == 0 ? string.Empty : "_reverse";
            var inputWeights = RequireFloatTensor(state, $"weight_ih_l0{suffix}", prefix);
            var recurrentWeights = RequireFloatTensor(state, $"weight_hh_l0{suffix}", prefix);
            var inputBias = RequireFloatTensor(state, $"bias_ih_l0{suffix}", prefix);
            var recurrentBias = RequireFloatTensor(state, $"bias_hh_l0{suffix}", prefix);
            AppendGateReordered(inputWeights, hiddenSize, inputSize, wValues);
            AppendGateReordered(recurrentWeights, hiddenSize, hiddenSize, rValues);
            AppendGateReordered(inputBias, hiddenSize, 1, bValues);
            AppendGateReordered(recurrentBias, hiddenSize, 1, bValues);
        }

        AddDerived("__oxy_W", [new CompilerFixedDimension(directions), new CompilerFixedDimension(4L * hiddenSize), new CompilerFixedDimension(inputSize)], wValues);
        AddDerived("__oxy_R", [new CompilerFixedDimension(directions), new CompilerFixedDimension(4L * hiddenSize), new CompilerFixedDimension(hiddenSize)], rValues);
        AddDerived("__oxy_B", [new CompilerFixedDimension(directions), new CompilerFixedDimension(8L * hiddenSize)], bValues);

        void AddDerived(string name, IEnumerable<CompilerDimension> dimensions, IEnumerable<CompilerScalarLiteral> values)
        {
            var tensor = new CompilerTensorLiteral(CompilerElementType.Float32, dimensions, values);
            members.Add(new CompilerTorchSharpStateMemberDescriptor(
                prefix + name,
                CompilerStateMemberKind.Initializer,
                new CompilerTensorType(CompilerElementType.Float32, tensor.Dimensions),
                tensor,
                "global::TorchSharp.torch.Tensor"));
        }
    }

    private static int ReadIntField(global::TorchSharp.torch.nn.Module module, string name)
    {
        var value = module.GetType().GetField(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)?.GetValue(module);
        return value is null
            ? throw new NotSupportedException($"TorchSharp module '{module.GetType().FullName}' has no integer setting '{name}'.")
            : Convert.ToInt32(value, System.Globalization.CultureInfo.InvariantCulture);
    }

    private static bool ReadBoolField(global::TorchSharp.torch.nn.Module module, string name) =>
        (bool?)module.GetType().GetField(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)?.GetValue(module)
        ?? throw new NotSupportedException($"TorchSharp module '{module.GetType().FullName}' has no Boolean setting '{name}'.");

    private static CompilerTensorLiteral RequireFloatTensor(
        IReadOnlyDictionary<string, global::TorchSharp.torch.Tensor> state,
        string name,
        string prefix)
    {
        if (!state.TryGetValue(name, out var tensor) || tensor.dtype != global::TorchSharp.torch.ScalarType.Float32)
        {
            throw new NotSupportedException($"TorchSharp LSTM state '{prefix}{name}' must be a Float32 tensor.");
        }

        return CreateTensorLiteral(tensor);
    }

    private static void AppendGateReordered(
        CompilerTensorLiteral tensor,
        int hiddenSize,
        int rowWidth,
        ICollection<CompilerScalarLiteral> destination)
    {
        if (tensor.Values.Count != checked(4 * hiddenSize * rowWidth))
        {
            throw new NotSupportedException("TorchSharp LSTM weight or bias shape does not match the configured hidden size.");
        }

        // PyTorch stores gates as input, forget, cell, output; ONNX expects input, output, forget, cell.
        foreach (var gate in new[] { 0, 3, 1, 2 })
        {
            var start = gate * hiddenSize * rowWidth;
            for (var index = 0; index < hiddenSize * rowWidth; index++)
            {
                destination.Add(tensor.Values[start + index]);
            }
        }
    }

    private static IReadOnlyList<CompilerTorchSharpChildModuleDescriptor> GetChildModules(
        global::TorchSharp.torch.nn.Module module,
        HashSet<global::TorchSharp.torch.nn.Module> activeModules
    )
    {
        return module.GetType()
            .GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
            .Where(field => field.DeclaringType == module.GetType())
            .Where(x => typeof(global::TorchSharp.torch.nn.Module).IsAssignableFrom(x.FieldType))
            .Select(
                field =>
                {
                    var childModule = field.GetValue(module) as global::TorchSharp.torch.nn.Module;
                    var childDescriptor = childModule is null
                        ? null
                        : CreateDescriptor(
                            module: childModule,
                            methodName: "forward",
                            document: null,
                            activeModules: activeModules);
                    return new CompilerTorchSharpChildModuleDescriptor(
                        name: field.Name,
                        typeName: field.FieldType.FullName ?? field.FieldType.Name,
                        blockName: field.Name,
                        module: childDescriptor);
                })
            .ToArray();
    }

    private static bool TryCreateScalarLiteral(
        object? value,
        out CompilerLiteral literal,
        out CompilerType type
    )
    {
        switch (value)
        {
            case bool boolean:
                literal = new CompilerBooleanLiteral(boolean);
                type = new CompilerScalarType(CompilerElementType.Boolean);
                return true;
            case int integer:
                literal = new CompilerSignedIntegerLiteral(CompilerElementType.Int32, integer);
                type = new CompilerScalarType(CompilerElementType.Int32);
                return true;
            case long integer:
                literal = new CompilerSignedIntegerLiteral(CompilerElementType.Int64, integer);
                type = new CompilerScalarType(CompilerElementType.Int64);
                return true;
            case float floating:
                literal = new CompilerFloatingPointLiteral(CompilerElementType.Float32, floating);
                type = new CompilerScalarType(CompilerElementType.Float32);
                return true;
            case double floating:
                literal = new CompilerFloatingPointLiteral(CompilerElementType.Float64, floating);
                type = new CompilerScalarType(CompilerElementType.Float64);
                return true;
            case string text:
                literal = new CompilerStringLiteral(text);
                type = new CompilerScalarType(CompilerElementType.String);
                return true;
            default:
                literal = null!;
                type = null!;
                return false;
        }
    }

    private static CompilerType TensorType(
        global::TorchSharp.torch.ScalarType? dataType,
        IEnumerable<TensorDimension>? dimensions
    )
    {
        var compilerDimensions = dimensions?.Select(
            x => x.Value switch
            {
                long value when value >= 0 => (CompilerDimension)new CompilerFixedDimension(value),
                string value => new CompilerSymbolicDimension(value),
                _ => new CompilerUnknownDimension(),
            });
        return new CompilerTensorType(
            ToCompilerElementType(dataType?.ToString()),
            compilerDimensions);
    }

    private static CompilerElementType ToCompilerElementType(string? scalarType)
    {
        return scalarType switch
        {
            "Bool" => CompilerElementType.Boolean,
            "Byte" or "UInt8" => CompilerElementType.UInt8,
            "Int8" => CompilerElementType.Int8,
            "Int16" => CompilerElementType.Int16,
            "UInt16" => CompilerElementType.UInt16,
            "Int32" => CompilerElementType.Int32,
            "UInt32" => CompilerElementType.UInt32,
            "Int64" => CompilerElementType.Int64,
            "UInt64" => CompilerElementType.UInt64,
            "Float16" => CompilerElementType.Float16,
            "BFloat16" => CompilerElementType.BFloat16,
            "Float64" => CompilerElementType.Float64,
            _ => CompilerElementType.Float32,
        };
    }
}

/// <summary>Extension form of <see cref="TorchSharpCompilerAdapter.CreateSource"/>.</summary>
public static class TorchModuleCompilerExtensions
{
    public static CSharpTorchSharpSource CreateCompilerSource(
        this global::TorchSharp.torch.nn.Module module,
        string methodName = "forward",
        string? document = null
    )
    {
        return TorchSharpCompilerAdapter.CreateSource(module, methodName, document);
    }
}
