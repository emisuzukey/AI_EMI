package ai.certifai.solution.convolution.mnist;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.filter.FilterInvalidValues;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.MinMaxSerializerStrategy;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class Q1Classifier {

    public static File inputFile;
    final private static int seed = 1234;
    final private static int epoch = 5;
    final private static int batchSize = 1000;

    public static void main(String[] args) throws IOException, InterruptedException {

        inputFile = new ClassPathResource("datavec/train.csv").getFile();
        CSVRecordReader rr = new CSVRecordReader(1, ',');
        rr.initialize(new FileSplit(inputFile));

        Schema schemaTrain = new Schema.Builder()
                .addColumnsInteger("ID","age")
                .addColumnCategorical("job", Arrays.asList("admin.", "blue-collar",
                        "entrepreneur", "housemaid","management","retired", "self-employed",
                        "services", "student","technician","unemployed", "unknown"))
                .addColumnCategorical("marital", Arrays.asList("married","divorced","single"))
                .addColumnCategorical("education", Arrays.asList("unknown","secondary","tertiary", "primary"))
                .addColumnCategorical("default", Arrays.asList("no", "yes"))
                .addColumnsInteger("balance")
                .addColumnCategorical("housing", Arrays.asList("no","yes"))
                .addColumnCategorical("loan", Arrays.asList("no","yes"))
                .addColumnCategorical("contact", Arrays.asList("telephone","cellular","unknown"))
                .addColumnsInteger("day")
                .addColumnCategorical("month", Arrays.asList("jan","feb","mar","apr","may","jun",
                        "jul","aug","sep","oct","nov","dec"))
                .addColumnsInteger("duration", "campaign", "pdays", "previous")
                .addColumnCategorical("poutcome", Arrays.asList("unknown","success","failure","other"))
                .addColumnCategorical("subscribed", Arrays.asList("no", "yes"))
                .build();
        System.out.println("original schema : \n " + schemaTrain);

        TransformProcess tpTrain = new TransformProcess.Builder(schemaTrain)
                .categoricalToInteger("job", "marital","education", "default","housing",
                        "loan","contact","month","poutcome","subscribed")
                .filter(new FilterInvalidValues())
                .build();

        List<List<Writable>> originalData = new ArrayList<>();

        while(rr.hasNext())
        {
            originalData.add(rr.next());
        }

        List<List<Writable>> transformedDataTrain = LocalTransformExecutor.execute(originalData, tpTrain);

        System.out.println(tpTrain.getFinalSchema());
        System.out.println(originalData.size());
        System.out.println(transformedDataTrain.size());

        CollectionRecordReader crr = new CollectionRecordReader(transformedDataTrain);
        RecordReaderDataSetIterator dataIterator = new RecordReaderDataSetIterator(crr, transformedDataTrain.size(), 17, 2);
        DataSet dataSet = dataIterator.next();


        //setting arrays
        SplitTestAndTrain testTrainSplit = dataSet.splitTestAndTrain(0.8);

        DataSet trainingSet = testTrainSplit.getTrain();
        DataSet validationSet = testTrainSplit.getTest();
        trainingSet.setLabelNames(Arrays.asList("0","1"));
        validationSet.setLabelNames(Arrays.asList("0","1"));

        DataNormalization scaler = new NormalizerMinMaxScaler();
        scaler.fit(trainingSet);
        scaler.transform(trainingSet);
        scaler.transform(validationSet);

        ViewIterator trainIter = new ViewIterator(trainingSet, batchSize);
        ViewIterator testIter = new ViewIterator(validationSet, batchSize);

        HashMap<Integer, Double> scheduler = new HashMap<>();
        scheduler.put(0, 1e-3);
        scheduler.put(2, 1e-4);
        scheduler.put(3, 1e-5);

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(new MapSchedule(ScheduleType.EPOCH, scheduler)))
                .activation(Activation.RELU)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(trainIter.inputColumns())
                        .nOut(256)
                        .build())
                .layer(new BatchNormalization())
                .layer(new DenseLayer.Builder()
                        .nOut(512)
                        .build())
                .layer(new BatchNormalization())
                .layer(new DenseLayer.Builder()
                        .nOut(512)
                        .build())
                .layer(new BatchNormalization())
                .layer(new DenseLayer.Builder()
                        .nOut(512)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .activation(Activation.SIGMOID)
                        .nOut(trainIter.totalOutcomes())
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        InMemoryStatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);
        model.setListeners(new StatsListener(storage), new ScoreIterationListener(100));

        ArrayList<Double> trainLoss = new ArrayList<>();
        ArrayList<Double> validLoss = new ArrayList<>();
        DataSetLossCalculator trainLossCalc = new DataSetLossCalculator(trainIter, true);
        DataSetLossCalculator validLossCalc = new DataSetLossCalculator(testIter, true);

        for (int i = 0; i <= epoch ; i++) {

            model.fit(trainIter);
            trainLoss.add(trainLossCalc.calculateScore(model));
            validLoss.add(validLossCalc.calculateScore(model));
        }

        Evaluation trainEval = model.evaluate(trainIter);
        Evaluation testEval = model.evaluate(testIter);

        System.out.println(trainEval.stats());
        System.out.println(testEval.stats());

        ModelSerializer.writeModel(model,"D:\\BACKUPDATA\\AI_SKYMIND\\midterm\\deposit.zip", true);
        NormalizerSerializer normalizerSerializer = new NormalizerSerializer().addStrategy(new MinMaxSerializerStrategy());
        normalizerSerializer.write(scaler,"D:\\BACKUPDATA\\AI_SKYMIND\\midterm\\normalizer.zip");

        List<List<Writable>> valCollection = RecordConverter.toRecords(validationSet);
        INDArray valArray = RecordConverter.toMatrix(DataType.FLOAT, valCollection);
        INDArray valFeature = valArray.getColumns(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16);

        List<String> prediction = model.predict(validationSet);
        INDArray output = model.output(valFeature);

        for (int i = 0; i < 10; i++) {

            System.out.println("Prediction : " + prediction.get(i) + "; Output: " + output.getRow(i));

        }


    }


}
