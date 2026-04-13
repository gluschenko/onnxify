using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Onnxify.Examples.Models;
using TorchSharp;
using static TorchSharp.torch;

namespace Onnxify.Examples
{
    internal static class ModelEvaluator
    {
        public static async Task<EvaluationResult> EvaluateTorch(
            AlexNet model,
            DataReader reader,
            int batchSize,
            Device device
        )
        {
            var confusionMatrix = new int[(int)reader.ClassCount, (int)reader.ClassCount];

            model.eval();

            await foreach (var batch in reader.BatchAsync(batchSize))
            {
                using var d = torch.NewDisposeScope();
                using var x = batch.GetDataTensor(device);
                using var output = model.forward(x);
                using var predicted = output.argmax(1).cpu();

                var predictions = predicted.data<long>().ToArray();
                UpdateConfusionMatrix(confusionMatrix, batch.LabelIndices, predictions);
            }

            return EvaluationResult.From(confusionMatrix);
        }

        public static async Task<EvaluationResult> EvaluateOnnx(
            string modelPath,
            DataReader reader,
            int batchSize
        )
        {
            var confusionMatrix = new int[(int)reader.ClassCount, (int)reader.ClassCount];

            using var session = new InferenceSession(modelPath);

            await foreach (var batch in reader.BatchAsync(batchSize))
            {
                using var d = torch.NewDisposeScope();
                using var x = batch.GetDataTensor(CPU);

                var inputData = x.data<float>().ToArray();
                var inputTensor = new DenseTensor<float>(
                    inputData,
                    [batch.Size, (int)x.shape[1], (int)x.shape[2], (int)x.shape[3]]
                );

                var inputValue = NamedOnnxValue.CreateFromTensor("input", inputTensor);
                var inputs = new[] { inputValue };

                using var results = session.Run(inputs);
                var outputTensor = results.Single().AsTensor<float>();
                var predictions = ArgMax(outputTensor, (int)reader.ClassCount);

                UpdateConfusionMatrix(confusionMatrix, batch.LabelIndices, predictions);
            }

            return EvaluationResult.From(confusionMatrix);
        }

        public static void PrintConfusionMatrix(
            string title,
            EvaluationResult result,
            IReadOnlyList<string> labelNames
        )
        {
            Console.WriteLine();
            Console.WriteLine(title);
            Console.WriteLine($"Accuracy: {result.Accuracy:0.000000}");

            var columnWidth = Math.Max(
                8,
                labelNames
                    .Append("actual \\ pred")
                    .Max(static x => x.Length) + 2
            );

            Console.Write("".PadRight(columnWidth));
            foreach (var label in labelNames)
            {
                Console.Write(label.PadLeft(columnWidth));
            }

            Console.WriteLine();

            for (var actual = 0; actual < labelNames.Count; actual++)
            {
                Console.Write(labelNames[actual].PadRight(columnWidth));
                for (var predicted = 0; predicted < labelNames.Count; predicted++)
                {
                    Console.Write(result.ConfusionMatrix[actual, predicted].ToString().PadLeft(columnWidth));
                }

                Console.WriteLine();
            }
        }

        private static int[] ArgMax(Tensor<float> outputTensor, int classCount)
        {
            var values = outputTensor.ToArray();
            var batchSize = values.Length / classCount;
            var predictions = new int[batchSize];

            for (var i = 0; i < batchSize; i++)
            {
                var bestClass = 0;
                var bestScore = float.MinValue;

                for (var classIndex = 0; classIndex < classCount; classIndex++)
                {
                    var score = values[i * classCount + classIndex];
                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestClass = classIndex;
                    }
                }

                predictions[i] = bestClass;
            }

            return predictions;
        }

        private static void UpdateConfusionMatrix(
            int[,] confusionMatrix,
            IReadOnlyList<int> labels,
            IReadOnlyList<long> predictions
        )
        {
            for (var i = 0; i < labels.Count; i++)
            {
                confusionMatrix[labels[i], (int)predictions[i]]++;
            }
        }

        private static void UpdateConfusionMatrix(
            int[,] confusionMatrix,
            IReadOnlyList<int> labels,
            IReadOnlyList<int> predictions
        )
        {
            for (var i = 0; i < labels.Count; i++)
            {
                confusionMatrix[labels[i], predictions[i]]++;
            }
        }
    }

    internal sealed record EvaluationResult(
        int[,] ConfusionMatrix,
        int Correct,
        int Total
    )
    {
        public float Accuracy => Total == 0 ? 0f : (float)Correct / Total;

        public static EvaluationResult From(int[,] confusionMatrix)
        {
            var correct = 0;
            var total = 0;

            for (var actual = 0; actual < confusionMatrix.GetLength(0); actual++)
            {
                for (var predicted = 0; predicted < confusionMatrix.GetLength(1); predicted++)
                {
                    var count = confusionMatrix[actual, predicted];
                    total += count;

                    if (actual == predicted)
                    {
                        correct += count;
                    }
                }
            }

            return new EvaluationResult(confusionMatrix, correct, total);
        }
    }
}
