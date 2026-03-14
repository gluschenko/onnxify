using Onnx;
using System;
using System.Collections.Generic;
using static Onnx.OnnxMlReflection;

namespace Onnxify
{
    internal abstract class OnnxNode
    {
        public abstract void AddAttribute(string argName, double value);
        public abstract void AddAttribute(string argName, long value);
        public abstract void AddAttribute(string argName, ReadOnlyMemory<char> value);
        public abstract void AddAttribute(string argName, string value);
        public abstract void AddAttribute(string argName, bool value);

        public abstract void AddAttribute(string argName, IEnumerable<double> value);
        public abstract void AddAttribute(string argName, IEnumerable<float> value);
        public abstract void AddAttribute(string argName, IEnumerable<long> value);
        public abstract void AddAttribute(string argName, IEnumerable<ReadOnlyMemory<char>> value);
        public abstract void AddAttribute(string argName, string[] value);
        public abstract void AddAttribute(string argName, IEnumerable<string> value);
        public abstract void AddAttribute(string argName, IEnumerable<bool> value);
        public abstract void AddAttribute(string argName, Type t);
    }

    internal sealed class OnnxNodeImpl : OnnxNode
    {
        private readonly NodeProto _node;

        public OnnxNodeImpl(NodeProto node)
        {
            _node = node;
        }

        public override void AddAttribute(string argName, double value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public override void AddAttribute(string argName, IEnumerable<double> value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public override void AddAttribute(string argName, IEnumerable<float> value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public override void AddAttribute(string argName, IEnumerable<bool> value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public override void AddAttribute(string argName, long value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public override void AddAttribute(string argName, IEnumerable<long> value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public override void AddAttribute(string argName, ReadOnlyMemory<char> value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public override void AddAttribute(string argName, string[] value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public override void AddAttribute(string argName, IEnumerable<ReadOnlyMemory<char>> value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public override void AddAttribute(string argName, IEnumerable<string> value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public override void AddAttribute(string argName, string value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public override void AddAttribute(string argName, bool value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public override void AddAttribute(string argName, Type value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
    }
}
