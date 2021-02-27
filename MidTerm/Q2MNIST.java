/**
EDITTED BY YOURS TRULY BUT DONT TRUST ME SINCE THIS IS NOT YET PERFECT
**/


package ai.certifai.training.convolution.mnist;

import javafx.application.Application;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;
public abstract class Q2MNIST extends Application {

    private static final Logger log = LoggerFactory.getLogger(Q2MNIST.class);


    private static final int height = 28;
    private static final int width = 28;
    private static final int channels = 1; // single channel for grayscale images
    private static final int outputNum = 10; // 10 digits classification
    private static final int batchSize = 3000;
    private static final int nEpochs = 1;
    private static final double learningRate = 0.001;
    private static MultiLayerNetwork model = null;

    private static final int seed = 1234;

    public static void main(String[] args) throws Exception {

        // define csv file location
        File inputFile = new ClassPathResource("datavec/mnist_784_csv.csv").getFile();
        FileSplit fileSplit = new FileSplit(inputFile);

        // get dataset using record reader. CSVRecordReader handles loading/parsing
        RecordReader recordReader = new CSVRecordReader(1, ',');
        recordReader.initialize(fileSplit);

        // create iterator from record reader
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, 784, 10);
        DataSet allData = iterator.next();
        System.out.println("batch size :" +batchSize);
        System.out.println("no of epoch :" +nEpochs);
        System.out.println("image size  : 28x28");
        System.out.println("Shape of allData vector:");
        System.out.println(Arrays.toString(allData.getFeatures().shape()));

        // shuffle and split all data into training and test set
        allData.shuffle();
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.8);
        org.nd4j.linalg.dataset.DataSet trainingData = testAndTrain.getTrain();
        org.nd4j.linalg.dataset.DataSet testData = testAndTrain.getTest();

        System.out.println("\nShape of training vector:");
        System.out.println(Arrays.toString(trainingData.getFeatures().shape()));
        System.out.println("\nShape of test vector:");
        System.out.println(Arrays.toString(testData.getFeatures().shape()));

        // create iterator for splitted training and test dataset
        DataSetIterator trainIterator = new ViewIterator(trainingData, 4);
        DataSetIterator testIterator = new ViewIterator(testData, 2);

        // normalize data to 0 - 1
        DataNormalization scaler = new NormalizerMinMaxScaler(0, 1);
        scaler.fit(trainIterator);
        trainIterator.setPreProcessor(scaler);
        testIterator.setPreProcessor(scaler);





    /*
   #### LAB STEP 1 #####
   Model configuration
    */
        log.info("Network configuration and training...");

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(1e-3))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(100).build())
                .layer(3, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(80).build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(60).build())
                .layer(5, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(40).build())
                .layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(height, width, 1)) // InputType.convolutional for normal image
                .backpropType(BackpropType.Standard)
                .build();

        model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        // evaluation while training (the score should go down)
        for (int i = 0; i < nEpochs; i++) {
            model.fit(trainIterator);

            log.info("Completed epoch {}", i);
            Evaluation eval = model.evaluate(testIterator);
            log.info(eval.stats());
            trainIterator.reset();
            testIterator.reset();
        }

    }
}