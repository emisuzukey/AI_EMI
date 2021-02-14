package ai.certifai.training.classification.transferlearning;

import ai.certifai.training.classification.transferlearning.Q3Test;
import org.datavec.image.transform.*;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Q3Train {

    private static final Logger log = org.slf4j.LoggerFactory.getLogger(ai.certifai.training.classification.transferlearning.Q3Train.class);

    private static int epochs = 10; //120
    private static int batchSize = 100;
    private static int seed = 1234;
    private static int numClasses =6;

    private static int height = 150;
    private static int width = 150;
    private static int channel = 3;

    private static final Random randNumGen = new Random(seed);

    public static void main(String[] args) throws Exception{

        // image augmentation
        ImageTransform horizontalFlip = new FlipImageTransform(1);
        ImageTransform cropImage = new CropImageTransform(25);
        ImageTransform rotateImage = new RotateImageTransform(randNumGen, 15);
        boolean shuffle = false;
        List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(
                new Pair<>(horizontalFlip,0.5),
                new Pair<>(rotateImage, 0.5),
                new Pair<>(cropImage,0.3)
        );

        ImageTransform transform = new PipelineImageTransform(pipeline,shuffle);

        ai.certifai.training.classification.transferlearning.Q3Test.setup(batchSize, 80, transform);

        //create iterators
        DataSetIterator trainIter = ai.certifai.training.classification.transferlearning.Q3Test.trainIterator();

        DataSetIterator testIter = Q3Test.testIterator();

        //model configuration
        double nonZeroBias = 0.1;
        double dropOut = 0.5;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(1e-3))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(channel)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .nIn(20)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(1024).build())
                .layer(5, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(1024).build())
                .layer(6, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(512).build())

                .layer(7, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(height, width, 1)) // InputType.convolutional for normal image
                .backpropType(BackpropType.Standard)
                .build();

        //train model and eval model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        log.info(model.summary());

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(
                new StatsListener( statsStorage),
                new ScoreIterationListener(100),
                new EvaluativeListener(trainIter, 1, InvocationType.EPOCH_END),
                new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END)
        );

        model.fit(trainIter, epochs);
        model.evaluate(testIter);
        Evaluation evalTest = model.evaluate(testIter);
        Evaluation evalTrain = model.evaluate(trainIter);
        System.out.println("train evaluate :" +evalTrain);
        System.out.println("test evaluate :" +evalTest);

    }
}
