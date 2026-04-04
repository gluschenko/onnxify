using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Onnxify.Examples.Models
{
    internal class AlexNetTrainer
    {
        private readonly AlexNet _model;
        private readonly DataReader _reader;

        public AlexNetTrainer(AlexNet model, DataReader reader)
        {
            _model = model;
            _reader = reader;
        }

        public void Train(
            int epochs = 5,
            int batchSize = 32,
            float learningRate = 1e-3f,
            int schedulerStepSize = 5,
            float schedulerGamma = 0.5f,
            float minLearningRate = 1e-5f,
            Device? device = null
        )
        {
            device ??= torch.cuda.is_available() ? CUDA : CPU;

            _model.to(device);
            _model.train();

            var optimizer = optim.Adam(_model.parameters(), learningRate);
            var criterion = CrossEntropyLoss();
            var scheduler = new StepLearningRateScheduler(
                learningRate,
                schedulerStepSize,
                schedulerGamma,
                minLearningRate
            );

            for (var epoch = 1; epoch <= epochs; epoch++)
            {
                var currentLearningRate = scheduler.GetLearningRate(epoch);
                scheduler.Apply(optimizer, epoch);

                Console.WriteLine($"Epoch {epoch}/{epochs} | lr {FormatLearningRate(currentLearningRate)}");

                var batchIndex = 0;
                var processedSamples = 0;
                var correctPredictions = 0;

                foreach (var batch in _reader.GetTrainingBatches(batchSize, shuffle: true))
                {
                    TrainBatch(
                        batch,
                        optimizer,
                        criterion,
                        device,
                        epoch,
                        epochs,
                        currentLearningRate,
                        ref batchIndex,
                        ref processedSamples,
                        ref correctPredictions
                    );
                }

                Console.WriteLine();
            }
        }

        private void TrainBatch(
            DataReader.Batch batch,
            optim.Optimizer optimizer,
            Loss<torch.Tensor, torch.Tensor, torch.Tensor> criterion,
            Device device,
            int epoch,
            int epochs,
            float learningRate,
            ref int batchIndex,
            ref int processedSamples,
            ref int correctPredictions
        )
        {
            using var d = torch.NewDisposeScope();
            using var x = torch.stack(batch.Data).to(device);
            using var y = torch.stack(batch.Labels).to(device).view(-1);

            optimizer.zero_grad();

            using var output = _model.forward(x);
            using var loss = criterion.call(output, y);

            loss.backward();
            optimizer.step();

            using var predicted = output.argmax(1);
            using var correct = predicted.eq(y);

            processedSamples += batch.LabelIndices.Length;
            correctPredictions += correct.sum().ToInt32();

            var accuracy = processedSamples == 0
                ? 0f
                : (float)correctPredictions / processedSamples;

            Console.Write(
                $"\rTrain: epoch {epoch}/{epochs} | batch {batchIndex + 1} | samples {processedSamples} | loss {loss.ToSingle():0.000000} | acc {accuracy:0.000000} | lr {FormatLearningRate(learningRate)}"
            );

            batchIndex++;
        }

        private static string FormatLearningRate(float learningRate)
        {
            return learningRate.ToString("0.######E+0");
        }

        private sealed class StepLearningRateScheduler
        {
            private readonly float _initialLearningRate;
            private readonly int _stepSize;
            private readonly float _gamma;
            private readonly float _minLearningRate;

            public StepLearningRateScheduler(
                float initialLearningRate,
                int stepSize,
                float gamma,
                float minLearningRate
            )
            {
                _initialLearningRate = initialLearningRate;
                _stepSize = Math.Max(1, stepSize);
                _gamma = gamma;
                _minLearningRate = MathF.Max(0f, minLearningRate);
            }

            public float GetLearningRate(int epoch)
            {
                var decaySteps = Math.Max(0, (epoch - 1) / _stepSize);
                var learningRate = _initialLearningRate * MathF.Pow(_gamma, decaySteps);
                return MathF.Max(_minLearningRate, learningRate);
            }

            public void Apply(optim.Optimizer optimizer, int epoch)
            {
                var learningRate = GetLearningRate(epoch);
                var paramGrous = optimizer.ParamGroups;
                foreach (var group in paramGrous)
                {
                    group.LearningRate = learningRate;
                }
            }
        }
    }
}
