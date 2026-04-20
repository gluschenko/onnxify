using System.Globalization;
using System.Runtime.InteropServices;
using Onnxify.Safetensors;
using static TorchSharp.torch;

namespace Onnxify.TorchSharp;

public readonly record struct TorchStateLoadResult(
    IReadOnlyList<string> Missing,
    IReadOnlyList<string> Unexpected);

public static class TorchSafetensors
{
    public static byte[] Save(
        IReadOnlyDictionary<string, Tensor> tensors,
        IReadOnlyDictionary<string, string>? metadata = null)
    {
        ArgumentNullException.ThrowIfNull(tensors);

        var serialized = FlattenForSave(tensors, forceContiguous: false, rejectSharedTensors: true);
        return global::Onnxify.Safetensors.Safetensors.Serialize(serialized, metadata);
    }

    public static void SaveFile(
        IReadOnlyDictionary<string, Tensor> tensors,
        string filename,
        IReadOnlyDictionary<string, string>? metadata = null)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        ArgumentNullException.ThrowIfNull(filename);

        var directory = Path.GetDirectoryName(filename);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var serialized = FlattenForSave(tensors, forceContiguous: false, rejectSharedTensors: true);
        global::Onnxify.Safetensors.Safetensors.SerializeToFile(serialized, metadata, filename);
    }

    public static Dictionary<string, Tensor> Load(
        byte[] data,
        Device? device = null)
    {
        ArgumentNullException.ThrowIfNull(data);

        var safetensors = global::Onnxify.Safetensors.Safetensors.Deserialize(data);
        var result = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach (var name in safetensors.Names())
        {
            result[name] = CreateTensor(safetensors.Tensor(name), device);
        }

        return result;
    }

    public static Dictionary<string, Tensor> LoadFile(
        string filename,
        Device? device = null)
    {
        ArgumentNullException.ThrowIfNull(filename);
        return Load(File.ReadAllBytes(filename), device);
    }

    public static void SaveModel(
        TorchModule module,
        string filename,
        IReadOnlyDictionary<string, string>? metadata = null,
        bool forceContiguous = true)
    {
        ArgumentNullException.ThrowIfNull(module);
        ArgumentNullException.ThrowIfNull(filename);

        var state = SnapshotStateDict(module);
        var toRemoves = RemoveDuplicateNames(state);
        var mergedMetadata = metadata is null
            ? null
            : new Dictionary<string, string>(metadata, StringComparer.Ordinal);

        foreach (var (keptName, toRemoveGroup) in toRemoves)
        {
            foreach (var toRemove in toRemoveGroup)
            {
                mergedMetadata ??= new Dictionary<string, string>(StringComparer.Ordinal);
                if (!mergedMetadata.ContainsKey(toRemove))
                {
                    mergedMetadata[toRemove] = keptName;
                }

                state.Remove(toRemove);
            }
        }

        try
        {
            SaveFileCore(state, filename, mergedMetadata, forceContiguous, rejectSharedTensors: false);
        }
        catch (ArgumentException ex) when (!forceContiguous)
        {
            throw new ArgumentException(
                $"{ex.Message} Or use SaveModel(..., forceContiguous: true), read the docs for potential caveats.",
                nameof(filename),
                ex);
        }
    }

    public static TorchStateLoadResult LoadModel(
        TorchModule module,
        string filename,
        bool strict = true,
        Device? device = null)
    {
        ArgumentNullException.ThrowIfNull(module);
        ArgumentNullException.ThrowIfNull(filename);

        using var loadedState = new TensorDictionaryScope(LoadFile(filename, device));
        var modelState = SnapshotStateDict(module);
        var toRemoves = RemoveDuplicateNames(modelState, preferredNames: loadedState.Names);

        var missing = new HashSet<string>(modelState.Keys, StringComparer.Ordinal);
        var unexpected = new HashSet<string>(StringComparer.Ordinal);

        foreach (var (name, sourceTensor) in loadedState.Tensors)
        {
            if (!modelState.TryGetValue(name, out var targetTensor))
            {
                unexpected.Add(name);
                continue;
            }

            ValidateStateTensorShape(name, targetTensor, sourceTensor);
            CopyTensorData(targetTensor, sourceTensor);
            missing.Remove(name);
        }

        foreach (var toRemoveGroup in toRemoves.Values)
        {
            foreach (var toRemove in toRemoveGroup)
            {
                if (!missing.Remove(toRemove))
                {
                    unexpected.Add(toRemove);
                }
            }
        }

        var missingList = missing.OrderBy(static x => x, StringComparer.Ordinal).ToArray();
        var unexpectedList = unexpected.OrderBy(static x => x, StringComparer.Ordinal).ToArray();

        if (strict && (missingList.Length > 0 || unexpectedList.Length > 0))
        {
            throw new InvalidOperationException(
                FormatLoadStateError(module.GetType().Name, missingList, unexpectedList));
        }

        return new TorchStateLoadResult(missingList, unexpectedList);
    }

    private static void SaveFileCore(
        IReadOnlyDictionary<string, Tensor> tensors,
        string filename,
        IReadOnlyDictionary<string, string>? metadata,
        bool forceContiguous,
        bool rejectSharedTensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        ArgumentNullException.ThrowIfNull(filename);

        var directory = Path.GetDirectoryName(filename);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var serialized = FlattenForSave(tensors, forceContiguous, rejectSharedTensors);
        global::Onnxify.Safetensors.Safetensors.SerializeToFile(serialized, metadata, filename);
    }

    private static KeyValuePair<string, TensorView>[] FlattenForSave(
        IReadOnlyDictionary<string, Tensor> tensors,
        bool forceContiguous,
        bool rejectSharedTensors)
    {
        EvaluateTensorsForSave(tensors, rejectSharedTensors);

        var flattened = new KeyValuePair<string, TensorView>[tensors.Count];
        var index = 0;
        foreach (var (name, tensor) in tensors)
        {
            flattened[index++] = new KeyValuePair<string, TensorView>(
                name,
                CreateTensorView(name, tensor, forceContiguous));
        }

        return flattened;
    }

    private static void EvaluateTensorsForSave(
        IReadOnlyDictionary<string, Tensor> tensors,
        bool rejectSharedTensors)
    {
        var sparseTensors = new List<string>();
        foreach (var (name, tensor) in tensors)
        {
            if (tensor is null || tensor.IsInvalid)
            {
                throw new ArgumentException($"Key `{name}` is invalid, expected a valid Torch tensor.");
            }

            if (!IsStridedTensor(tensor))
            {
                sparseTensors.Add(name);
            }
        }

        if (sparseTensors.Count > 0)
        {
            throw new ArgumentException(
                $"You are trying to save a sparse tensors: `{string.Join(", ", sparseTensors)}` which this library does not support." +
                " You can make it a dense tensor before saving with `.to_dense()` but be aware this might" +
                " make a much larger file than needed.");
        }

        if (!rejectSharedTensors)
        {
            return;
        }

        var sharedPointers = FindSharedTensors(tensors);
        var failing = sharedPointers.Where(static names => names.Count > 1).ToArray();
        if (failing.Length > 0)
        {
            throw new InvalidOperationException(
                "Some tensors share memory, this will lead to duplicate memory on disk and potential differences when loading them again: " +
                $"[{string.Join(", ", failing.Select(static names => $"[{string.Join(", ", names.OrderBy(static x => x, StringComparer.Ordinal))}]"))}]. " +
                "A potential way to correctly save your model is to use `SaveModel`. " +
                "More information at https://huggingface.co/docs/safetensors/torch_shared_tensors");
        }
    }

    private static TensorView CreateTensorView(string name, Tensor tensor, bool forceContiguous)
    {
        using var detached = tensor.detach();

        if (!forceContiguous && !detached.is_contiguous())
        {
            throw new ArgumentException(
                $"You are trying to save a non contiguous tensor: `{name}` which is not allowed. It either means you" +
                " are trying to save tensors which are reference of each other in which case it's recommended to save" +
                " only the full tensors, and reslice at load time, or simply call `.contiguous()` on your tensor to" +
                " pack it before saving.");
        }

        using var cpuTensor = detached.device.type.ToString().Equals("CPU", StringComparison.OrdinalIgnoreCase)
            ? detached.alias()
            : detached.cpu();
        using var readyTensor = forceContiguous ? cpuTensor.contiguous() : cpuTensor.alias();

        if (!readyTensor.is_contiguous())
        {
            throw new ArgumentException(
                $"You are trying to save a non contiguous tensor: `{name}` which is not allowed. It either means you" +
                " are trying to save tensors which are reference of each other in which case it's recommended to save" +
                " only the full tensors, and reslice at load time, or simply call `.contiguous()` on your tensor to" +
                " pack it before saving.");
        }

        var dataType = MapTorchDataType(readyTensor);
        var shape = readyTensor.shape.Select(static x => checked((ulong)x)).ToArray();
        var bytes = CopyRawTensorBytes(readyTensor, checked((int)GetTensorByteLength(readyTensor)));
        return new TensorView(dataType, shape, bytes);
    }

    private static Tensor CreateTensor(TensorView tensorView, Device? device)
    {
        var shape = tensorView.Shape.Select(static x => checked((long)x)).ToArray();
        var scalarType = GetScalarType(tensorView.DataType);

        if (tensorView.Data.IsEmpty)
        {
            return device is null
                ? empty(shape, dtype: scalarType)
                : empty(shape, dtype: scalarType, device: device);
        }

        var raw = tensorView.Data.ToArray();
        var bufferTensor = frombuffer(raw, dtype: scalarType);
        try
        {
            var reshaped = bufferTensor.reshape(shape);
            try
            {
                var materialized = reshaped.clone();
                if (device is null)
                {
                    return materialized;
                }

                var moved = materialized.to(device);
                materialized.Dispose();
                return moved;
            }
            finally
            {
                reshaped.Dispose();
            }
        }
        finally
        {
            bufferTensor.Dispose();
        }
    }

    private static Dictionary<string, Tensor> SnapshotStateDict(TorchModule module)
    {
        var state = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach (var (name, tensor) in module.state_dict())
        {
            if (tensor is not null && !tensor.IsInvalid)
            {
                state[name] = tensor;
            }
        }

        return state;
    }

    private static Dictionary<string, List<string>> RemoveDuplicateNames(
        IReadOnlyDictionary<string, Tensor> stateDict,
        IReadOnlyCollection<string>? preferredNames = null,
        IReadOnlyCollection<string>? discardNames = null)
    {
        var preferred = preferredNames is null
            ? new HashSet<string>(StringComparer.Ordinal)
            : new HashSet<string>(preferredNames, StringComparer.Ordinal);
        var discarded = discardNames is null
            ? new HashSet<string>(StringComparer.Ordinal)
            : new HashSet<string>(discardNames, StringComparer.Ordinal);

        var sharedTensors = FindSharedTensors(stateDict);
        var toRemove = new Dictionary<string, List<string>>(StringComparer.Ordinal);

        foreach (var shared in sharedTensors)
        {
            if (shared.Count < 2)
            {
                continue;
            }

            var completeNames = shared
                .Where(name => IsCompleteTensor(stateDict[name]))
                .ToHashSet(StringComparer.Ordinal);

            if (completeNames.Count == 0)
            {
                throw new InvalidOperationException(
                    "Error while trying to find names to remove to save state dict, but found no suitable name to keep" +
                    $" for saving amongst: [{string.Join(", ", shared.OrderBy(static x => x, StringComparer.Ordinal))}]." +
                    " None is covering the entire storage. Refusing to save/load the model since you could be storing much more memory than needed." +
                    " Please refer to https://huggingface.co/docs/safetensors/torch_shared_tensors for more information.");
            }

            var keepName = completeNames.OrderBy(static x => x, StringComparer.Ordinal).First();

            var preferredFromModel = completeNames.Except(discarded, StringComparer.Ordinal)
                .OrderBy(static x => x, StringComparer.Ordinal)
                .ToArray();
            if (preferredFromModel.Length > 0)
            {
                keepName = preferredFromModel[0];
            }

            var preferredFromFile = preferred.Intersect(completeNames, StringComparer.Ordinal)
                .OrderBy(static x => x, StringComparer.Ordinal)
                .ToArray();
            if (preferredFromFile.Length > 0)
            {
                keepName = preferredFromFile[0];
            }

            foreach (var name in shared.OrderBy(static x => x, StringComparer.Ordinal))
            {
                if (name == keepName)
                {
                    continue;
                }

                if (!toRemove.TryGetValue(keepName, out var group))
                {
                    group = [];
                    toRemove[keepName] = group;
                }

                group.Add(name);
            }
        }

        return toRemove;
    }

    private static List<HashSet<string>> FindSharedTensors(IReadOnlyDictionary<string, Tensor> stateDict)
    {
        var grouped = new Dictionary<StorageDescriptor, HashSet<string>>();

        foreach (var (name, tensor) in stateDict)
        {
            if (!TryGetStorageDescriptor(tensor, out var descriptor))
            {
                continue;
            }

            if (!grouped.TryGetValue(descriptor, out var names))
            {
                names = new HashSet<string>(StringComparer.Ordinal);
                grouped.Add(descriptor, names);
            }

            names.Add(name);
        }

        return FilterSharedNotShared(grouped.Values.ToList(), stateDict);
    }

    private static List<HashSet<string>> FilterSharedNotShared(
        List<HashSet<string>> tensors,
        IReadOnlyDictionary<string, Tensor> stateDict)
    {
        var filtered = new List<HashSet<string>>();

        foreach (var shared in tensors)
        {
            if (shared.Count < 2)
            {
                filtered.Add(shared);
                continue;
            }

            var areas = shared
                .Select(name =>
                {
                    var tensor = stateDict[name];
                    return GetTensorMemoryArea(name, tensor);
                })
                .OrderBy(static x => x.Start)
                .ThenBy(static x => x.Stop)
                .ToArray();

            var lastStop = areas[0].Stop;
            filtered.Add([areas[0].Name]);
            foreach (var area in areas.Skip(1))
            {
                if (area.Start >= lastStop)
                {
                    filtered.Add([area.Name]);
                }
                else
                {
                    filtered[^1].Add(area.Name);
                }

                lastStop = Math.Max(lastStop, area.Stop);
            }
        }

        return filtered;
    }

    private static bool IsCompleteTensor(Tensor tensor)
    {
        if (!TryGetStorageDescriptor(tensor, out var descriptor))
        {
            return true;
        }

        return GetTensorDataPointer(tensor) == descriptor.Pointer
            && GetTensorByteLength(tensor) == descriptor.SizeInBytes;
    }

    private static bool TryGetStorageDescriptor(Tensor tensor, out StorageDescriptor descriptor)
    {
        descriptor = default;

        if (tensor is null || tensor.IsInvalid || tensor.device.type.ToString().Equals("META", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        try
        {
            using var storage = tensor.storage<byte>();
            var storagePointer = storage.data_ptr().ToInt64();
            var storageSize = storage.nbytes();

            if (storagePointer == 0 || storageSize == 0)
            {
                return false;
            }

            var deviceKey = Convert.ToString(tensor.device, CultureInfo.InvariantCulture)
                ?? tensor.device.type.ToString();

            descriptor = new StorageDescriptor(deviceKey, storagePointer, storageSize);
            return true;
        }
        catch (NotImplementedException)
        {
            return false;
        }
    }

    private static bool IsStridedTensor(Tensor tensor)
    {
        return !tensor.is_sparse();
    }

    private static byte[] CopyRawTensorBytes(Tensor tensor, int byteLength)
    {
        var pointer = GetTensorDataPointer(tensor);
        if (pointer == 0)
        {
            return new byte[byteLength];
        }

        var bytes = new byte[byteLength];
        if (byteLength > 0)
        {
            Marshal.Copy(new IntPtr(pointer), bytes, 0, byteLength);
        }

        return bytes;
    }

    private static long GetTensorDataPointer(Tensor tensor)
    {
        if (tensor is null || tensor.IsInvalid)
        {
            return 0;
        }

        try
        {
            using var storage = tensor.storage<byte>();
            var storagePointer = storage.data_ptr().ToInt64();
            if (storagePointer == 0)
            {
                return 0;
            }

            var offsetInBytes = checked(tensor.storage_offset() * tensor.ElementSize);
            return checked(storagePointer + offsetInBytes);
        }
        catch (NotImplementedException)
        {
            return 0;
        }
    }

    private static ulong GetTensorByteLength(Tensor tensor)
    {
        if (tensor is null || tensor.IsInvalid)
        {
            return 0;
        }

        try
        {
            return checked((ulong)tensor.NumberOfElements * (ulong)tensor.ElementSize);
        }
        catch (OverflowException)
        {
            throw new InvalidOperationException(
                $"Overflow computing byte length for torch tensor type {MapTorchDataType(tensor).ToWireName()}.");
        }
    }

    private static void ValidateStateTensorShape(string name, Tensor target, Tensor source)
    {
        var targetShape = target.shape.Select(static x => checked((long)x)).ToArray();
        var sourceShape = source.shape.Select(static x => checked((long)x)).ToArray();
        if (!targetShape.SequenceEqual(sourceShape))
        {
            throw new InvalidOperationException(
                $"Shape mismatch for tensor '{name}'. Model expects [{string.Join(", ", targetShape)}] but file contains [{string.Join(", ", sourceShape)}].");
        }
    }

    private static void CopyTensorData(Tensor target, Tensor source)
    {
        using var detachedTarget = target.detach();
        using var detachedSource = source.detach();
        using var sourceOnTargetDevice = detachedTarget.device.Equals(detachedSource.device)
            ? detachedSource.alias()
            : detachedSource.to(detachedTarget.device);

        detachedTarget.copy_(sourceOnTargetDevice);
    }

    private static string FormatLoadStateError(
        string moduleName,
        IReadOnlyList<string> missing,
        IReadOnlyList<string> unexpected)
    {
        var error = $"Error(s) in loading state_dict for {moduleName}:";
        if (missing.Count > 0)
        {
            error += $"\n    Missing key(s) in state_dict: {string.Join(", ", missing.Select(static key => $"\"{key}\""))}";
        }

        if (unexpected.Count > 0)
        {
            error += $"\n    Unexpected key(s) in state_dict: {string.Join(", ", unexpected.Select(static key => $"\"{key}\""))}";
        }

        return error;
    }

    private static DataType MapTorchDataType(Tensor tensor)
    {
        var dataTypeName = Convert.ToString(tensor.dtype, CultureInfo.InvariantCulture);
        return dataTypeName switch
        {
            "Float32" or "Float" => DataType.F32,
            "Float64" or "Double" => DataType.F64,
            "Float16" or "Half" => DataType.F16,
            "BFloat16" => DataType.Bf16,
            "Int64" => DataType.I64,
            "Int32" => DataType.I32,
            "Int16" => DataType.I16,
            "Int8" => DataType.I8,
            "UInt8" or "Byte" => DataType.U8,
            "UInt16" => DataType.U16,
            "UInt32" => DataType.U32,
            "UInt64" => DataType.U64,
            "Bool" => DataType.Bool,
            "ComplexFloat32" or "Complex32" or "ComplexFloat" => DataType.C64,
            _ => throw new NotSupportedException(
                $"Unsupported Torch tensor data type for safetensors: {dataTypeName}."),
        };
    }

    private static ScalarType GetScalarType(DataType dataType)
    {
        var scalarTypeName = dataType switch
        {
            DataType.F64 => "Float64",
            DataType.F32 => "Float32",
            DataType.F16 => "Float16",
            DataType.Bf16 => "BFloat16",
            DataType.I64 => "Int64",
            DataType.I32 => "Int32",
            DataType.I16 => "Int16",
            DataType.I8 => "Int8",
            DataType.U8 => "UInt8",
            DataType.U16 => "UInt16",
            DataType.U32 => "UInt32",
            DataType.U64 => "UInt64",
            DataType.Bool => "Bool",
            DataType.C64 => "ComplexFloat32",
            _ => throw new NotSupportedException(
                $"Unsupported safetensors data type for Torch tensor materialization: {dataType.ToWireName()}."),
        };

        if (!Enum.TryParse<ScalarType>(scalarTypeName, ignoreCase: false, out var scalarType))
        {
            throw new NotSupportedException(
                $"TorchSharp does not expose ScalarType '{scalarTypeName}' required for safetensors type {dataType.ToWireName()}.");
        }

        return scalarType;
    }

    private static TensorMemoryArea GetTensorMemoryArea(string name, Tensor tensor)
    {
        var start = GetTensorDataPointer(tensor);
        if (start == 0 || tensor.NumberOfElements == 0)
        {
            return new TensorMemoryArea(start, start, name);
        }

        var shape = tensor.shape;
        var strides = tensor.stride();
        var elementSize = tensor.ElementSize;
        long minRelativeOffset = 0;
        long maxRelativeOffset = 0;

        for (var i = 0; i < shape.Length; i++)
        {
            var dimension = shape[i];
            if (dimension <= 1)
            {
                continue;
            }

            var extent = checked((dimension - 1) * strides[i]);
            if (extent >= 0)
            {
                maxRelativeOffset = checked(maxRelativeOffset + extent);
            }
            else
            {
                minRelativeOffset = checked(minRelativeOffset + extent);
            }
        }

        var areaStart = checked(start + (minRelativeOffset * elementSize));
        var areaStop = checked(start + ((maxRelativeOffset + 1) * elementSize));
        return new TensorMemoryArea(areaStart, areaStop, name);
    }

    private readonly record struct StorageDescriptor(string Device, long Pointer, ulong SizeInBytes);
    private readonly record struct TensorMemoryArea(long Start, long Stop, string Name);

    private sealed class TensorDictionaryScope : IDisposable
    {
        private readonly Dictionary<string, Tensor> _tensors;

        public TensorDictionaryScope(Dictionary<string, Tensor> tensors)
        {
            _tensors = tensors;
        }

        public IEnumerable<KeyValuePair<string, Tensor>> Tensors => _tensors;

        public IReadOnlyCollection<string> Names => _tensors.Keys;

        public void Dispose()
        {
            foreach (var tensor in _tensors.Values)
            {
                tensor.Dispose();
            }
        }
    }
}
