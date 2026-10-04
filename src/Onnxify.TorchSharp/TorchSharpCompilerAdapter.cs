using System.Reflection;
using Onnxify.Compiler;

namespace Onnxify.TorchSharp;

/// <summary>
/// Creates compiler-neutral C# TorchSharp sources from runtime modules. Decompilation and syntax
/// scanning remain in <see cref="Onnxify.Compiler.Compiler"/>.
/// </summary>
public static class TorchSharpCompilerAdapter
{
    /// <summary>Creates a compiler source for the module's <c>forward</c> method.</summary>
    public static CSharpTorchSharpSource CreateSource(
        global::TorchSharp.torch.nn.Module module,
        string methodName = "forward",
        string? document = null
    )
    {
        ArgumentNullException.ThrowIfNull(module);
        ArgumentException.ThrowIfNullOrWhiteSpace(methodName);

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
        var stateMembers = GetStateMembers(module);
        var childModules = GetChildModules(module);
        var helpers = moduleType
            .GetMethods(BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)
            .Where(x => !string.Equals(x.Name, methodName, StringComparison.Ordinal))
            .Where(x => x.DeclaringType == moduleType)
            .Where(x => !x.IsSpecialName)
            .OrderBy(x => x.MetadataToken)
            .Select(x => new CompilerTorchSharpHelperMethodDescriptor(x.Name))
            .GroupBy(x => x.Name, StringComparer.Ordinal)
            .Select(x => x.First())
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
        return new CSharpTorchSharpSource(descriptor);
    }

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
                        attribute is null ? TensorType(null, null) : TensorType(attribute.DataType, attribute.Dimensions),
                        parameter.ParameterType.FullName ?? "global::TorchSharp.torch.Tensor");
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
                        method.ReturnType.FullName ?? "global::TorchSharp.torch.Tensor"))
                .ToArray();
        }

        return [
            new CompilerTorchSharpValueDescriptor(
                "output",
                TensorType(null, null),
                method.ReturnType.FullName ?? "global::TorchSharp.torch.Tensor"),
        ];
    }

    private static IReadOnlyList<CompilerTorchSharpStateMemberDescriptor> GetStateMembers(
        global::TorchSharp.torch.nn.Module module
    )
    {
        var members = new List<CompilerTorchSharpStateMemberDescriptor>();
        foreach (var field in module.GetType().GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic))
        {
            var value = field.GetValue(module);
            if (value is global::TorchSharp.torch.Tensor tensor)
            {
                members.Add(
                    new CompilerTorchSharpStateMemberDescriptor(
                        field.Name,
                        CompilerStateMemberKind.Buffer,
                        TensorType(tensor.dtype, tensor.GetShape().Select(x => new TensorDimension(x))),
                        csharpTypeName: "global::TorchSharp.torch.Tensor"));
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

        return members
            .GroupBy(x => x.Name, StringComparer.Ordinal)
            .Select(x => x.First())
            .ToArray();
    }

    private static IReadOnlyList<CompilerTorchSharpChildModuleDescriptor> GetChildModules(
        global::TorchSharp.torch.nn.Module module
    )
    {
        return module.GetType()
            .GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
            .Where(x => typeof(global::TorchSharp.torch.nn.Module).IsAssignableFrom(x.FieldType))
            .Select(
                x => new CompilerTorchSharpChildModuleDescriptor(
                    x.Name,
                    x.FieldType.FullName ?? x.FieldType.Name))
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
