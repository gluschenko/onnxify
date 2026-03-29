using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Onnxify.Examples
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
            Device? device = null
        )
        {
            device ??= torch.cuda.is_available() ? CUDA : CPU;

            _model.to(device);
            _model.train();

            var optimizer = optim.Adam(_model.parameters(), learningRate);
            var criterion = CrossEntropyLoss();
            var scheduler = new StepLearningRateScheduler(learningRate, schedulerStepSize, schedulerGamma);

            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                var currentLearningRate = scheduler.GetLearningRate(epoch);
                scheduler.Apply(optimizer, epoch);

                Console.WriteLine($"Epoch {epoch}/{epochs} | lr {currentLearningRate:0.000000}");

                var batchData = new List<Tensor>();
                var batchLabels = new List<Tensor>();

                var batchIndex = 0;
                var processedSamples = 0;
                var correctPredictions = 0;

                foreach (var (data, label) in _reader.Data())
                {
                    batchData.Add(data);
                    batchLabels.Add(label);

                    if (batchData.Count == batchSize)
                    {
                        TrainBatch(
                            batchData,
                            batchLabels,
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

                        batchData.Clear();
                        batchLabels.Clear();
                    }
                }

                if (batchData.Count > 0)
                {
                    TrainBatch(
                        batchData,
                        batchLabels,
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
            List<Tensor> batchData,
            List<Tensor> batchLabels,
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
            using var x = torch.stack(batchData).to(device); // [N, C, H, W]
            using var y = torch.stack(batchLabels).to(device).view(-1); // [N]

            optimizer.zero_grad();

            using var output = _model.forward(x);
            using var loss = criterion.call(output, y);

            loss.backward();
            optimizer.step();

            using var predicted = output.argmax(1);
            using var correct = predicted.eq(y);

            processedSamples += batchData.Count;
            correctPredictions += correct.sum().ToInt32();
            var accuracy = processedSamples == 0
                ? 0f
                : (float)correctPredictions / processedSamples;

            Console.Write(
                $"\rTrain: epoch {epoch}/{epochs} | batch {batchIndex + 1} | samples {processedSamples} | loss {loss.ToSingle():0.000000} | acc {accuracy:0.000000} | lr {learningRate:0.000000}"
            );

            batchIndex++;
        }

        private sealed class StepLearningRateScheduler
        {
            private readonly float _initialLearningRate;
            private readonly int _stepSize;
            private readonly float _gamma;

            public StepLearningRateScheduler(float initialLearningRate, int stepSize, float gamma)
            {
                _initialLearningRate = initialLearningRate;
                _stepSize = Math.Max(1, stepSize);
                _gamma = gamma;
            }

            public float GetLearningRate(int epoch)
            {
                var decaySteps = Math.Max(0, (epoch - 1) / _stepSize);
                return _initialLearningRate * MathF.Pow(_gamma, decaySteps);
            }

            public void Apply(optim.Optimizer optimizer, int epoch)
            {
                var learningRate = GetLearningRate(epoch);
                foreach (var x in optimizer.ParamGroups)
                {
                    x.LearningRate = learningRate;
                }
            }
        }
    }
}
