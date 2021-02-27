package ai.certifai.training.segmentation.asl;

import ai.certifai.Helper;
import ai.certifai.solution.segmentation.CustomLabelGenerator;
import ai.certifai.solution.segmentation.car.CarDataSetIterator;
import ai.certifai.utilities.DataUtilities;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class DataIterator {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(DataIterator.class);
    private static final int height = 224;
    private static final int width = 224;
    private static final int channels = 1;
    private static final long seed = 12345;
    private static final Random random = new Random(seed);
    private static String inputDir;
    private static String downloadLink;
    private static InputSplit trainData, valData;
    private static int batchSize;

    private static List<Pair<String, String>> replacement = Arrays.asList(
            new Pair<>("inputs", "masks_png"),
            new Pair<>(".jpg", "_mask.png")
    );
    private static ai.certifai.solution.segmentation.CustomLabelGenerator labelMaker = new CustomLabelGenerator(height, width, channels, replacement);

    //scale input to 0 - 1
    private static DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    private static ImageTransform transform;

    public DataIterator() {
    }

    //This method instantiates an ImageRecordReader and subsequently a RecordReaderDataSetIterator based on it
    private static RecordReaderDataSetIterator makeIterator(InputSplit split) throws IOException {

        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
//        Both train and val iterator need the preprocessing of converting RGB to Grayscale
        recordReader.initialize(split, transform);
        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, 1, true);
        iter.setPreProcessor(scaler);

        return iter;
    }

    public static RecordReaderDataSetIterator trainIterator() throws IOException {
        return makeIterator(trainData);
    }

    public static RecordReaderDataSetIterator valIterator() throws IOException {
        return makeIterator(valData);
    }

    public static void setup(int batchSizeArg, double trainPerc, ImageTransform imageTransform) throws IOException {
        transform = imageTransform;
        setup(batchSizeArg, trainPerc);
    }

    public static void setup(int batchSizeArg, double trainPerc) throws IOException {

        downloadLink = Helper.getPropValues("https://drive.google.com/file/d/1RMNgRReYsD_X9upsIc7QE3iejj69M_d-/view?usp=sharing");
        inputDir = Paths.get(
                System.getProperty("user.home"),
                Helper.getPropValues("dl4j_home.data")
        ).toString();

        File dataZip = new File(Paths.get(inputDir, "sign-language.zip").toString());
        File classFolder = new File(Paths.get(inputDir, "sign-language").toString());

        if (!dataZip.exists()) {
            log.info(String.format("Downloading %s from %s.", dataZip.getAbsolutePath(), downloadLink));
            DataUtilities.downloadFile(downloadLink, dataZip.getAbsolutePath());
        }

        if (!classFolder.exists()) {
            log.info(String.format("Extracting %s into %s.", dataZip.getAbsolutePath(), classFolder.getAbsolutePath()));

            if(!Helper.getCheckSum(dataZip.getAbsolutePath())
                    .equalsIgnoreCase(Helper.getPropValues("dataset.segmentationCar.hash"))){
                System.out.println("Downloaded file is incomplete");
                System.exit(0);
            }

            DataUtilities.extractZip(dataZip.getAbsolutePath(), classFolder.getAbsolutePath());
        }

        batchSize = batchSizeArg;

        inputDir = Paths.get(
                System.getProperty("user.home"),
                Helper.getPropValues("dl4j_home.data")
        ).toString();


        File imagesPath = new File(Paths.get(inputDir, "sign-language", "sign-language", "train", "inputs").toString());
        FileSplit imageFileSplit = new FileSplit(imagesPath, NativeImageLoader.ALLOWED_FORMATS, random);

        BalancedPathFilter imageSplitPathFilter = new BalancedPathFilter(random, NativeImageLoader.ALLOWED_FORMATS, labelMaker);
        InputSplit[] imagesSplits = imageFileSplit.sample(imageSplitPathFilter, trainPerc, 1 - trainPerc);

        trainData = imagesSplits[0];
        valData = imagesSplits[1];
    }
}
