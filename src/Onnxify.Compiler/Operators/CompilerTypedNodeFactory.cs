namespace Onnxify.Compiler.Operators
{
    internal static class CompilerTypedNodeFactory
    {
        private delegate OnnxNode CompilerNodeFactory(
            string name,
            IReadOnlyList<IOnnxGraphEdge?> inputs,
            IReadOnlyList<IOnnxGraphEdge?> outputs,
            IReadOnlyList<OnnxAttribute> attributes
        );

        private static readonly IReadOnlyDictionary<Type, CompilerNodeFactory> _factories =
            new Dictionary<Type, CompilerNodeFactory>
            {
                [typeof(Onnxify.Abs)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Abs(
                        name,
                        new Onnxify.AbsInputOutputOptions
                        {
                            X = inputs[0] ?? throw new InvalidOperationException("Missing required input 'X'"),
                            Y = outputs[0] ?? throw new InvalidOperationException("Missing required output 'Y'"),
                        }
                    );
                },
                [typeof(Onnxify.Acos)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Acos(
                        name,
                        new Onnxify.AcosInputOutputOptions
                        {
                            Input = inputs[0] ?? throw new InvalidOperationException("Missing required input 'input'"),
                            Output = outputs[0] ?? throw new InvalidOperationException("Missing required output 'output'"),
                        }
                    );
                },
                [typeof(Onnxify.Acosh)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Acosh(
                        name,
                        new Onnxify.AcoshInputOutputOptions
                        {
                            Input = inputs[0] ?? throw new InvalidOperationException("Missing required input 'input'"),
                            Output = outputs[0] ?? throw new InvalidOperationException("Missing required output 'output'"),
                        }
                    );
                },
                [typeof(Onnxify.Add)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Add(
                        name,
                        new Onnxify.AddInputOutputOptions
                        {
                            A = inputs[0] ?? throw new InvalidOperationException("Missing required input 'A'"),
                            B = inputs[1] ?? throw new InvalidOperationException("Missing required input 'B'"),
                            C = outputs[0] ?? throw new InvalidOperationException("Missing required output 'C'"),
                        }
                    );
                },
                [typeof(Onnxify.And)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.And(
                        name,
                        new Onnxify.AndInputOutputOptions
                        {
                            A = inputs[0] ?? throw new InvalidOperationException("Missing required input 'A'"),
                            B = inputs[1] ?? throw new InvalidOperationException("Missing required input 'B'"),
                            C = outputs[0] ?? throw new InvalidOperationException("Missing required output 'C'"),
                        }
                    );
                },
                [typeof(Onnxify.Asin)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Asin(
                        name,
                        new Onnxify.AsinInputOutputOptions
                        {
                            Input = inputs[0] ?? throw new InvalidOperationException("Missing required input 'input'"),
                            Output = outputs[0] ?? throw new InvalidOperationException("Missing required output 'output'"),
                        }
                    );
                },
                [typeof(Onnxify.Asinh)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Asinh(
                        name,
                        new Onnxify.AsinhInputOutputOptions
                        {
                            Input = inputs[0] ?? throw new InvalidOperationException("Missing required input 'input'"),
                            Output = outputs[0] ?? throw new InvalidOperationException("Missing required output 'output'"),
                        }
                    );
                },
                [typeof(Onnxify.Atan)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Atan(
                        name,
                        new Onnxify.AtanInputOutputOptions
                        {
                            Input = inputs[0] ?? throw new InvalidOperationException("Missing required input 'input'"),
                            Output = outputs[0] ?? throw new InvalidOperationException("Missing required output 'output'"),
                        }
                    );
                },
                [typeof(Onnxify.Atanh)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Atanh(
                        name,
                        new Onnxify.AtanhInputOutputOptions
                        {
                            Input = inputs[0] ?? throw new InvalidOperationException("Missing required input 'input'"),
                            Output = outputs[0] ?? throw new InvalidOperationException("Missing required output 'output'"),
                        }
                    );
                },
                [typeof(Onnxify.Cast)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Cast(
                        name,
                        new Onnxify.CastInputOutputOptions
                        {
                            Input = inputs[0] ?? throw new InvalidOperationException("Missing required input 'input'"),
                            RoundMode = (string?)attributes.GetValueOrDefault("round_mode"),
                            Saturate = (long?)attributes.GetValueOrDefault("saturate"),
                            To = (long)(attributes.GetValueOrDefault("to") ?? throw new InvalidOperationException($"Missing value 'to'")),
                            Output = outputs[0] ?? throw new InvalidOperationException("Missing required output 'output'"),
                        }
                    );
                },
                [typeof(Onnxify.Clip)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    return new Onnxify.Clip(
                        name,
                        new Onnxify.ClipInputOutputOptions
                        {
                            Input = inputs[0] ?? throw new InvalidOperationException("Missing required input 'input'"),
                            Min = inputs.Count > 1 && !string.IsNullOrEmpty(inputs[1]?.Name) ? inputs[1] : null,
                            Max = inputs.Count > 2 && !string.IsNullOrEmpty(inputs[2]?.Name) ? inputs[2] : null,
                            Output = outputs[0] ?? throw new InvalidOperationException("Missing required output 'output'"),
                        }
                    );
                },
                [typeof(Onnxify.Ceil)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Ceil(
                        name,
                        new Onnxify.CeilInputOutputOptions
                        {
                            X = inputs[0] ?? throw new InvalidOperationException("Missing required input 'X'"),
                            Y = outputs[0] ?? throw new InvalidOperationException("Missing required output 'Y'"),
                        }
                    );
                },
                [typeof(Onnxify.Celu)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Celu(
                        name,
                        new Onnxify.CeluInputOutputOptions
                        {
                            X = inputs[0] ?? throw new InvalidOperationException("Missing required input 'X'"),
                            Alpha = (float?)attributes.GetValueOrDefault("alpha"),
                            Y = outputs[0] ?? throw new InvalidOperationException("Missing required output 'Y'"),
                        }
                    );
                },
                [typeof(Onnxify.Conv)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Conv(
                        name,
                        new Onnxify.ConvInputOutputOptions
                        {
                            X = inputs[0] ?? throw new InvalidOperationException("Missing required input 'X'"),
                            W = inputs[1] ?? throw new InvalidOperationException("Missing required input 'W'"),
                            B = inputs.Count > 2 && !string.IsNullOrEmpty(inputs[2]?.Name) ? inputs[2] : null,
                            AutoPad = (string?)attributes.GetValueOrDefault("auto_pad"),
                            Dilations = (long[]?)attributes.GetValueOrDefault("dilations"),
                            Group = (long?)attributes.GetValueOrDefault("group"),
                            KernelShape = (long[]?)attributes.GetValueOrDefault("kernel_shape"),
                            Pads = (long[]?)attributes.GetValueOrDefault("pads"),
                            Strides = (long[]?)attributes.GetValueOrDefault("strides"),
                            Y = outputs[0] ?? throw new InvalidOperationException("Missing required output 'Y'"),
                        }
                    );
                },
                [typeof(Onnxify.Cos)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Cos(
                        name,
                        new Onnxify.CosInputOutputOptions
                        {
                            Input = inputs[0] ?? throw new InvalidOperationException("Missing required input 'input'"),
                            Output = outputs[0] ?? throw new InvalidOperationException("Missing required output 'output'"),
                        }
                    );
                },
                [typeof(Onnxify.Cosh)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Cosh(
                        name,
                        new Onnxify.CoshInputOutputOptions
                        {
                            Input = inputs[0] ?? throw new InvalidOperationException("Missing required input 'input'"),
                            Output = outputs[0] ?? throw new InvalidOperationException("Missing required output 'output'"),
                        }
                    );
                },
                [typeof(Onnxify.Div)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Div(
                        name,
                        new Onnxify.DivInputOutputOptions
                        {
                            A = inputs[0] ?? throw new InvalidOperationException("Missing required input 'A'"),
                            B = inputs[1] ?? throw new InvalidOperationException("Missing required input 'B'"),
                            C = outputs[0] ?? throw new InvalidOperationException("Missing required output 'C'"),
                        }
                    );
                },
                [typeof(Onnxify.Elu)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Elu(
                        name,
                        new Onnxify.EluInputOutputOptions
                        {
                            X = inputs[0] ?? throw new InvalidOperationException("Missing required input 'X'"),
                            Alpha = (float?)attributes.GetValueOrDefault("alpha"),
                            Y = outputs[0] ?? throw new InvalidOperationException("Missing required output 'Y'"),
                        }
                    );
                },
                [typeof(Onnxify.Equal)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Equal(
                        name,
                        new Onnxify.EqualInputOutputOptions
                        {
                            A = inputs[0] ?? throw new InvalidOperationException("Missing required input 'A'"),
                            B = inputs[1] ?? throw new InvalidOperationException("Missing required input 'B'"),
                            C = outputs[0] ?? throw new InvalidOperationException("Missing required output 'C'"),
                        }
                    );
                },
                [typeof(Onnxify.Erf)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Erf(
                        name,
                        new Onnxify.ErfInputOutputOptions
                        {
                            Input = inputs[0] ?? throw new InvalidOperationException("Missing required input 'input'"),
                            Output = outputs[0] ?? throw new InvalidOperationException("Missing required output 'output'"),
                        }
                    );
                },
                [typeof(Onnxify.Exp)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Exp(
                        name,
                        new Onnxify.ExpInputOutputOptions
                        {
                            Input = inputs[0] ?? throw new InvalidOperationException("Missing required input 'input'"),
                            Output = outputs[0] ?? throw new InvalidOperationException("Missing required output 'output'"),
                        }
                    );
                },
                [typeof(Onnxify.Floor)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Floor(
                        name,
                        new Onnxify.FloorInputOutputOptions
                        {
                            X = inputs[0] ?? throw new InvalidOperationException("Missing required input 'X'"),
                            Y = outputs[0] ?? throw new InvalidOperationException("Missing required output 'Y'"),
                        }
                    );
                },
                [typeof(Onnxify.Flatten)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Flatten(
                        name,
                        new Onnxify.FlattenInputOutputOptions
                        {
                            Input = inputs[0] ?? throw new InvalidOperationException("Missing required input 'input'"),
                            Axis = (long?)attributes.GetValueOrDefault("axis"),
                            Output = outputs[0] ?? throw new InvalidOperationException("Missing required output 'output'"),
                        }
                    );
                },
                [typeof(Onnxify.GlobalAveragePool)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    return new Onnxify.GlobalAveragePool(
                        name,
                        new Onnxify.GlobalAveragePoolInputOutputOptions
                        {
                            X = inputs[0] ?? throw new InvalidOperationException("Missing required input 'X'"),
                            Y = outputs[0] ?? throw new InvalidOperationException("Missing required output 'Y'"),
                        }
                    );
                },
                [typeof(Onnxify.Gelu)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Gelu(
                        name,
                        new Onnxify.GeluInputOutputOptions
                        {
                            X = inputs[0] ?? throw new InvalidOperationException("Missing required input 'X'"),
                            Approximate = (string?)attributes.GetValueOrDefault("approximate"),
                            Y = outputs[0] ?? throw new InvalidOperationException("Missing required output 'Y'"),
                        }
                    );
                },
                [typeof(Onnxify.Gemm)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Gemm(
                        name,
                        new Onnxify.GemmInputOutputOptions
                        {
                            A = inputs[0] ?? throw new InvalidOperationException("Missing required input 'A'"),
                            B = inputs[1] ?? throw new InvalidOperationException("Missing required input 'B'"),
                            C = inputs.Count > 2 && !string.IsNullOrEmpty(inputs[2]?.Name) ? inputs[2] : null,
                            Alpha = (float?)attributes.GetValueOrDefault("alpha"),
                            Beta = (float?)attributes.GetValueOrDefault("beta"),
                            TransA = (long?)attributes.GetValueOrDefault("transA"),
                            TransB = (long?)attributes.GetValueOrDefault("transB"),
                            Y = outputs[0] ?? throw new InvalidOperationException("Missing required output 'Y'"),
                        }
                    );
                },
                [typeof(Onnxify.Greater)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Greater(
                        name,
                        new Onnxify.GreaterInputOutputOptions
                        {
                            A = inputs[0] ?? throw new InvalidOperationException("Missing required input 'A'"),
                            B = inputs[1] ?? throw new InvalidOperationException("Missing required input 'B'"),
                            C = outputs[0] ?? throw new InvalidOperationException("Missing required output 'C'"),
                        }
                    );
                },
                [typeof(Onnxify.GreaterOrEqual)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.GreaterOrEqual(
                        name,
                        new Onnxify.GreaterOrEqualInputOutputOptions
                        {
                            A = inputs[0] ?? throw new InvalidOperationException("Missing required input 'A'"),
                            B = inputs[1] ?? throw new InvalidOperationException("Missing required input 'B'"),
                            C = outputs[0] ?? throw new InvalidOperationException("Missing required output 'C'"),
                        }
                    );
                },
                [typeof(Onnxify.HardSigmoid)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.HardSigmoid(
                        name,
                        new Onnxify.HardSigmoidInputOutputOptions
                        {
                            X = inputs[0] ?? throw new InvalidOperationException("Missing required input 'X'"),
                            Alpha = (float?)attributes.GetValueOrDefault("alpha"),
                            Beta = (float?)attributes.GetValueOrDefault("beta"),
                            Y = outputs[0] ?? throw new InvalidOperationException("Missing required output 'Y'"),
                        }
                    );
                },
                [typeof(Onnxify.HardSwish)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.HardSwish(
                        name,
                        new Onnxify.HardSwishInputOutputOptions
                        {
                            X = inputs[0] ?? throw new InvalidOperationException("Missing required input 'X'"),
                            Y = outputs[0] ?? throw new InvalidOperationException("Missing required output 'Y'"),
                        }
                    );
                },
                [typeof(Onnxify.IsNaN)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.IsNaN(
                        name,
                        new Onnxify.IsNaNInputOutputOptions
                        {
                            X = inputs[0] ?? throw new InvalidOperationException("Missing required input 'X'"),
                            Y = outputs[0] ?? throw new InvalidOperationException("Missing required output 'Y'"),
                        }
                    );
                },
                [typeof(Onnxify.IsInf)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.IsInf(
                        name,
                        new Onnxify.IsInfInputOutputOptions
                        {
                            X = inputs[0] ?? throw new InvalidOperationException("Missing required input 'X'"),
                            DetectNegative = (long?)attributes.GetValueOrDefault("detect_negative") ?? 1,
                            DetectPositive = (long?)attributes.GetValueOrDefault("detect_positive") ?? 1,
                            Y = outputs[0] ?? throw new InvalidOperationException("Missing required output 'Y'"),
                        }
                    );
                },
                [typeof(Onnxify.LeakyRelu)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.LeakyRelu(
                        name,
                        new Onnxify.LeakyReluInputOutputOptions
                        {
                            X = inputs[0] ?? throw new InvalidOperationException("Missing required input 'X'"),
                            Alpha = (float?)attributes.GetValueOrDefault("alpha"),
                            Y = outputs[0] ?? throw new InvalidOperationException("Missing required output 'Y'"),
                        }
                    );
                },
                [typeof(Onnxify.Less)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Less(
                        name,
                        new Onnxify.LessInputOutputOptions
                        {
                            A = inputs[0] ?? throw new InvalidOperationException("Missing required input 'A'"),
                            B = inputs[1] ?? throw new InvalidOperationException("Missing required input 'B'"),
                            C = outputs[0] ?? throw new InvalidOperationException("Missing required output 'C'"),
                        }
                    );
                },
                [typeof(Onnxify.LessOrEqual)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.LessOrEqual(
                        name,
                        new Onnxify.LessOrEqualInputOutputOptions
                        {
                            A = inputs[0] ?? throw new InvalidOperationException("Missing required input 'A'"),
                            B = inputs[1] ?? throw new InvalidOperationException("Missing required input 'B'"),
                            C = outputs[0] ?? throw new InvalidOperationException("Missing required output 'C'"),
                        }
                    );
                },
                [typeof(Onnxify.Log)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Log(
                        name,
                        new Onnxify.LogInputOutputOptions
                        {
                            Input = inputs[0] ?? throw new InvalidOperationException("Missing required input 'input'"),
                            Output = outputs[0] ?? throw new InvalidOperationException("Missing required output 'output'"),
                        }
                    );
                },
                [typeof(Onnxify.MatMul)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.MatMul(
                        name,
                        new Onnxify.MatMulInputOutputOptions
                        {
                            A = inputs[0] ?? throw new InvalidOperationException("Missing required input 'A'"),
                            B = inputs[1] ?? throw new InvalidOperationException("Missing required input 'B'"),
                            Y = outputs[0] ?? throw new InvalidOperationException("Missing required output 'Y'"),
                        }
                    );
                },
                [typeof(Onnxify.Max)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Max(
                        name,
                        new Onnxify.MaxInputOutputOptions
                        {
                            Data0 = inputs.Skip(0).OfType<IOnnxGraphEdge>().ToArray(),
                            OutputMax = outputs[0] ?? throw new InvalidOperationException("Missing required output 'max'"),
                        }
                    );
                },
                [typeof(Onnxify.Min)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Min(
                        name,
                        new Onnxify.MinInputOutputOptions
                        {
                            Data0 = inputs.Skip(0).OfType<IOnnxGraphEdge>().ToArray(),
                            OutputMin = outputs[0] ?? throw new InvalidOperationException("Missing required output 'min'"),
                        }
                    );
                },
                [typeof(Onnxify.Mish)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Mish(
                        name,
                        new Onnxify.MishInputOutputOptions
                        {
                            X = inputs[0] ?? throw new InvalidOperationException("Missing required input 'X'"),
                            Y = outputs[0] ?? throw new InvalidOperationException("Missing required output 'Y'"),
                        }
                    );
                },
                [typeof(Onnxify.Mod)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Mod(
                        name,
                        new Onnxify.ModInputOutputOptions
                        {
                            A = inputs[0] ?? throw new InvalidOperationException("Missing required input 'A'"),
                            B = inputs[1] ?? throw new InvalidOperationException("Missing required input 'B'"),
                            Fmod = (long?)attributes.GetValueOrDefault("fmod"),
                            C = outputs[0] ?? throw new InvalidOperationException("Missing required output 'C'"),
                        }
                    );
                },
                [typeof(Onnxify.Mul)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Mul(
                        name,
                        new Onnxify.MulInputOutputOptions
                        {
                            A = inputs[0] ?? throw new InvalidOperationException("Missing required input 'A'"),
                            B = inputs[1] ?? throw new InvalidOperationException("Missing required input 'B'"),
                            C = outputs[0] ?? throw new InvalidOperationException("Missing required output 'C'"),
                        }
                    );
                },
                [typeof(Onnxify.Neg)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Neg(
                        name,
                        new Onnxify.NegInputOutputOptions
                        {
                            X = inputs[0] ?? throw new InvalidOperationException("Missing required input 'X'"),
                            Y = outputs[0] ?? throw new InvalidOperationException("Missing required output 'Y'"),
                        }
                    );
                },
                [typeof(Onnxify.Not)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Not(
                        name,
                        new Onnxify.NotInputOutputOptions
                        {
                            X = inputs[0] ?? throw new InvalidOperationException("Missing required input 'X'"),
                            Y = outputs[0] ?? throw new InvalidOperationException("Missing required output 'Y'"),
                        }
                    );
                },
                [typeof(Onnxify.Or)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Or(
                        name,
                        new Onnxify.OrInputOutputOptions
                        {
                            A = inputs[0] ?? throw new InvalidOperationException("Missing required input 'A'"),
                            B = inputs[1] ?? throw new InvalidOperationException("Missing required input 'B'"),
                            C = outputs[0] ?? throw new InvalidOperationException("Missing required output 'C'"),
                        }
                    );
                },
                [typeof(Onnxify.Pow)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Pow(
                        name,
                        new Onnxify.PowInputOutputOptions
                        {
                            X = inputs[0] ?? throw new InvalidOperationException("Missing required input 'X'"),
                            Y = inputs[1] ?? throw new InvalidOperationException("Missing required input 'Y'"),
                            Z = outputs[0] ?? throw new InvalidOperationException("Missing required output 'Z'"),
                        }
                    );
                },
                [typeof(Onnxify.PRelu)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.PRelu(
                        name,
                        new Onnxify.PReluInputOutputOptions
                        {
                            X = inputs[0] ?? throw new InvalidOperationException("Missing required input 'X'"),
                            Slope = inputs[1] ?? throw new InvalidOperationException("Missing required input 'slope'"),
                            Y = outputs[0] ?? throw new InvalidOperationException("Missing required output 'Y'"),
                        }
                    );
                },
                [typeof(Onnxify.Reciprocal)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Reciprocal(
                        name,
                        new Onnxify.ReciprocalInputOutputOptions
                        {
                            X = inputs[0] ?? throw new InvalidOperationException("Missing required input 'X'"),
                            Y = outputs[0] ?? throw new InvalidOperationException("Missing required output 'Y'"),
                        }
                    );
                },
                [typeof(Onnxify.Relu)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Relu(
                        name,
                        new Onnxify.ReluInputOutputOptions
                        {
                            X = inputs[0] ?? throw new InvalidOperationException("Missing required input 'X'"),
                            Y = outputs[0] ?? throw new InvalidOperationException("Missing required output 'Y'"),
                        }
                    );
                },
                [typeof(Onnxify.Round)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Round(
                        name,
                        new Onnxify.RoundInputOutputOptions
                        {
                            X = inputs[0] ?? throw new InvalidOperationException("Missing required input 'X'"),
                            Y = outputs[0] ?? throw new InvalidOperationException("Missing required output 'Y'"),
                        }
                    );
                },
                [typeof(Onnxify.Selu)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Selu(
                        name,
                        new Onnxify.SeluInputOutputOptions
                        {
                            X = inputs[0] ?? throw new InvalidOperationException("Missing required input 'X'"),
                            Alpha = (float?)attributes.GetValueOrDefault("alpha"),
                            Gamma = (float?)attributes.GetValueOrDefault("gamma"),
                            Y = outputs[0] ?? throw new InvalidOperationException("Missing required output 'Y'"),
                        }
                    );
                },
                [typeof(Onnxify.Sigmoid)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Sigmoid(
                        name,
                        new Onnxify.SigmoidInputOutputOptions
                        {
                            X = inputs[0] ?? throw new InvalidOperationException("Missing required input 'X'"),
                            Y = outputs[0] ?? throw new InvalidOperationException("Missing required output 'Y'"),
                        }
                    );
                },
                [typeof(Onnxify.Sign)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Sign(
                        name,
                        new Onnxify.SignInputOutputOptions
                        {
                            Input = inputs[0] ?? throw new InvalidOperationException("Missing required input 'input'"),
                            Output = outputs[0] ?? throw new InvalidOperationException("Missing required output 'output'"),
                        }
                    );
                },
                [typeof(Onnxify.Sin)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Sin(
                        name,
                        new Onnxify.SinInputOutputOptions
                        {
                            Input = inputs[0] ?? throw new InvalidOperationException("Missing required input 'input'"),
                            Output = outputs[0] ?? throw new InvalidOperationException("Missing required output 'output'"),
                        }
                    );
                },
                [typeof(Onnxify.Sinh)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Sinh(
                        name,
                        new Onnxify.SinhInputOutputOptions
                        {
                            Input = inputs[0] ?? throw new InvalidOperationException("Missing required input 'input'"),
                            Output = outputs[0] ?? throw new InvalidOperationException("Missing required output 'output'"),
                        }
                    );
                },
                [typeof(Onnxify.Softplus)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Softplus(
                        name,
                        new Onnxify.SoftplusInputOutputOptions
                        {
                            X = inputs[0] ?? throw new InvalidOperationException("Missing required input 'X'"),
                            Y = outputs[0] ?? throw new InvalidOperationException("Missing required output 'Y'"),
                        }
                    );
                },
                [typeof(Onnxify.Softsign)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Softsign(
                        name,
                        new Onnxify.SoftsignInputOutputOptions
                        {
                            Input = inputs[0] ?? throw new InvalidOperationException("Missing required input 'input'"),
                            Output = outputs[0] ?? throw new InvalidOperationException("Missing required output 'output'"),
                        }
                    );
                },
                [typeof(Onnxify.Sqrt)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Sqrt(
                        name,
                        new Onnxify.SqrtInputOutputOptions
                        {
                            X = inputs[0] ?? throw new InvalidOperationException("Missing required input 'X'"),
                            Y = outputs[0] ?? throw new InvalidOperationException("Missing required output 'Y'"),
                        }
                    );
                },
                [typeof(Onnxify.Sub)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Sub(
                        name,
                        new Onnxify.SubInputOutputOptions
                        {
                            A = inputs[0] ?? throw new InvalidOperationException("Missing required input 'A'"),
                            B = inputs[1] ?? throw new InvalidOperationException("Missing required input 'B'"),
                            C = outputs[0] ?? throw new InvalidOperationException("Missing required output 'C'"),
                        }
                    );
                },
                [typeof(Onnxify.Swish)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Swish(
                        name,
                        new Onnxify.SwishInputOutputOptions
                        {
                            X = inputs[0] ?? throw new InvalidOperationException("Missing required input 'X'"),
                            Alpha = (float?)attributes.GetValueOrDefault("alpha"),
                            Y = outputs[0] ?? throw new InvalidOperationException("Missing required output 'Y'"),
                        }
                    );
                },
                [typeof(Onnxify.Tan)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Tan(
                        name,
                        new Onnxify.TanInputOutputOptions
                        {
                            Input = inputs[0] ?? throw new InvalidOperationException("Missing required input 'input'"),
                            Output = outputs[0] ?? throw new InvalidOperationException("Missing required output 'output'"),
                        }
                    );
                },
                [typeof(Onnxify.Tanh)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Tanh(
                        name,
                        new Onnxify.TanhInputOutputOptions
                        {
                            Input = inputs[0] ?? throw new InvalidOperationException("Missing required input 'input'"),
                            Output = outputs[0] ?? throw new InvalidOperationException("Missing required output 'output'"),
                        }
                    );
                },
                [typeof(Onnxify.ThresholdedRelu)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.ThresholdedRelu(
                        name,
                        new Onnxify.ThresholdedReluInputOutputOptions
                        {
                            X = inputs[0] ?? throw new InvalidOperationException("Missing required input 'X'"),
                            Alpha = (float?)attributes.GetValueOrDefault("alpha"),
                            Y = outputs[0] ?? throw new InvalidOperationException("Missing required output 'Y'"),
                        }
                    );
                },
                [typeof(Onnxify.Where)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Where(
                        name,
                        new Onnxify.WhereInputOutputOptions
                        {
                            Condition = inputs[0] ?? throw new InvalidOperationException("Missing required input 'condition'"),
                            X = inputs[1] ?? throw new InvalidOperationException("Missing required input 'X'"),
                            Y = inputs[2] ?? throw new InvalidOperationException("Missing required input 'Y'"),
                            Output = outputs[0] ?? throw new InvalidOperationException("Missing required output 'output'"),
                        }
                    );
                },
                [typeof(Onnxify.Transpose)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Transpose(
                        name,
                        new Onnxify.TransposeInputOutputOptions
                        {
                            Data = inputs[0] ?? throw new InvalidOperationException("Missing required input 'data'"),
                            Perm = attributes.TryGetValue("perm", out var permutation) ? (long[]?)permutation : null,
                            Transposed = outputs[0] ?? throw new InvalidOperationException("Missing required output 'transposed'"),
                        }
                    );
                },
                [typeof(Onnxify.Reshape)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Reshape(
                        name,
                        new Onnxify.ReshapeInputOutputOptions
                        {
                            Data = inputs[0] ?? throw new InvalidOperationException("Missing required input 'data'"),
                            Shape = inputs[1] ?? throw new InvalidOperationException("Missing required input 'shape'"),
                            Allowzero = (long?)attributes.GetValueOrDefault("allowzero"),
                            Reshaped = outputs[0] ?? throw new InvalidOperationException("Missing required output 'reshaped'"),
                        }
                    );
                },
                [typeof(Onnxify.Unsqueeze)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    return new Onnxify.Unsqueeze(
                        name,
                        new Onnxify.UnsqueezeInputOutputOptions
                        {
                            Data = inputs[0] ?? throw new InvalidOperationException("Missing required input 'data'"),
                            Axes = inputs[1] ?? throw new InvalidOperationException("Missing required input 'axes'"),
                            Expanded = outputs[0] ?? throw new InvalidOperationException("Missing required output 'expanded'"),
                        }
                    );
                },
                [typeof(Onnxify.Xor)] = static (name, inputs, outputs, onnxAttributes) =>
                {
                    var attributes = onnxAttributes.ToDictionary(attribute => attribute.Name, attribute => attribute.GetValue(), StringComparer.Ordinal);
                    return new Onnxify.Xor(
                        name,
                        new Onnxify.XorInputOutputOptions
                        {
                            A = inputs[0] ?? throw new InvalidOperationException("Missing required input 'A'"),
                            B = inputs[1] ?? throw new InvalidOperationException("Missing required input 'B'"),
                            C = outputs[0] ?? throw new InvalidOperationException("Missing required output 'C'"),
                        }
                    );
                },
            };

        public static TNode Create<TNode>(
            string name,
            IReadOnlyList<IOnnxGraphEdge?> inputs,
            IReadOnlyList<IOnnxGraphEdge?> outputs,
            IReadOnlyList<OnnxAttribute> onnxAttributes
        ) where TNode : OnnxNode
        {
            if (!_factories.TryGetValue(typeof(TNode), out var factory))
            {
                throw new InvalidOperationException($"No generated ONNX factory is registered for node type '{typeof(TNode).FullName}'.");
            }

            return factory(name, inputs, outputs, onnxAttributes) is TNode node
                ? node
                : throw new InvalidOperationException($"Generated factory returned an unexpected node type for '{typeof(TNode).FullName}'.");
        }
    }
}

