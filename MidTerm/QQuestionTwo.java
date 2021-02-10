package ai.certifai.solution.datavec.loadcsv;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

public class QQuestionTwo {
    private static File inputFile;
    private static int skipNumLines = 1;
    private static char delimiter = ',';
    final static int seed = 1234;
    final static int batchSize = 500;
    final static int epoch =10;

    public static void main(String[] args)  throws Exception {

        File inputFile = new ClassPathResource("datavec/mnist_784_csv.csv").getFile();
        RecordReader inputReader = new CSVRecordReader(skipNumLines, delimiter);
        inputReader.initialize(new FileSplit(inputFile));



        MnistDataSetIterator trainMinst = new MnistDataSetIterator(batchSize,true,seed);
        MnistDataSetIterator testMinst = new MnistDataSetIterator(batchSize,false,seed);

        NormalizerMinMaxScaler scaler = new NormalizerMinMaxScaler(0,1);
        DataSet trainSet = trainMinst.next();
        scaler.fit(trainSet);
        scaler.transform(trainSet);

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(1e-3))
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(trainMinst.inputColumns())
                        .nOut(124)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nOut(282)
                        .build())
                .layer(2, new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nOut(testMinst.totalOutcomes())
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);

        for(int i =0; i<= epoch; i++)
        {
            model.fit(testMinst);
        }

        Evaluation evalTrain = model.evaluate(trainMinst);
        Evaluation evaltest = model.evaluate(testMinst);

        System.out.println(evalTrain.stats());
        System.out.println(evaltest.stats());

    }

}
