package org.ml_methods_group.evaluation;

import org.ml_methods_group.cache.HashDatabase;
import org.ml_methods_group.classification.classifiers.CompositeClassifier;
import org.ml_methods_group.classification.classifiers.KNearestNeighbors;
import org.ml_methods_group.classification.classifiers.NearestCluster;
import org.ml_methods_group.common.*;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.common.extractors.BOWExtractor;
import org.ml_methods_group.common.serialization.MarkedSolutionsClusters;
import org.ml_methods_group.common.serialization.SolutionsDataset;
import org.ml_methods_group.evaluation.approaches.Approach;
import org.ml_methods_group.evaluation.approaches.BOWApproach;
import org.ml_methods_group.testing.BasicClassificationTester;
import org.ml_methods_group.testing.ClassificationTester;
import org.ml_methods_group.testing.ClassificationTestingResult;
import org.ml_methods_group.testing.validation.Validator;
import org.ml_methods_group.testing.validation.basic.PrecalculatedValidator;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import static org.ml_methods_group.common.Solution.Verdict.FAIL;
import static org.ml_methods_group.evaluation.ClusterizationEvaluation.createChangesExtractor;

public class FinalTester {

//    public static String problem = "min_max";
    public static String problem = "double_equality";
//    public static String problem = "loggers";
//    public static String problem = "deserialization";

//    public static String classifier = "closest-cluster";
        public static String classifier = "k-nearest-5";
//        public static String classifier = "k-nearest-10";
//        public static String classifier = "k-nearest-15";

    public static double hacThreshold = 0.4;
    public static int clusterSizeThreshold = 20;
    public static String clusteringApproach = "def_vec";


    public static void main(String[] args) throws Exception {
        SolutionsDataset train = SolutionsDataset.load(Paths.get("cache", "datasets", problem, "train.tmp"));
        final List<Solution> incorrect = SolutionsDataset
                .load(Paths.get("cache", "datasets", problem, "test.tmp"))
                .getValues(x -> x.getVerdict() == FAIL);
        Database database = new HashDatabase(EvaluationInfo.pathToCache);
        final FeaturesExtractor<Solution, Changes> changesExtractor = createChangesExtractor(train, database);
        System.out.println("Extractor created");
        long start = System.currentTimeMillis();

        final Approach<BOWExtractor.BOWVector> approach = BOWApproach.getDefaultApproach(20000, train, changesExtractor);
//        final Approach<List<String>> approach = JaccardApproach.getDefaultApproach(changesExtractor);
//        final Approach<List<String>> approach = JaccardApproach.getExtendedApproach(changesExtractor);
//        final Approach<List<String>> approach = JaccardApproach.getFullApproach(changesExtractor);
//        final Approach<Changes> approach = FuzzyJaccardApproach.getDefaultApproach(changesExtractor);
//        final Approach<List<double[]>> approach = VectorizationApproach.getDefaultApproach(train, changesExtractor);
//        final Approach<double[]> approach = VectorizationApproach.getSumApproach(train, changesExtractor);
//        final Approach<double[]> approach = VectorizationApproach.getMeanApproach(train, changesExtractor);

        final Validator<Solution, String> validator = loadValidator(problem);
        final Path path = Paths.get("cache", "marked_clusters", problem, clusteringApproach,
                hacThreshold + "_" + clusterSizeThreshold + ".mc");
        final MarkedSolutionsClusters clusters = MarkedSolutionsClusters.load(path);
        Classifier<Solution, String> errorClassifier = createClassifierForName(classifier, approach);
        errorClassifier.train(clusters);
        final ClassificationTester<Solution, String> tester = new BasicClassificationTester<>(incorrect, validator);
        final ClassificationTestingResult result1 = tester.test(errorClassifier);
        double[] t = {0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5};
        List<Combiner.Test> points = new ArrayList<>();
        for (double tr : t) {
            System.out.print(tr + " ");
            System.out.print(result1.getCoverage(tr) + ",");
            System.out.print(result1.getPrecision(tr) + ",");
            System.out.println(result1.getAccuracy());
            points.add(new Combiner.Test(result1.getPrecision(tr), result1.getCoverage(tr), tr));
        }
        System.out.println(getAUC(points));
    }

    private static Validator<Solution, String> loadValidator(String problem) throws IOException {
        final Path path = Paths.get("cache", "validators", problem + ".pvd");
        return PrecalculatedValidator.load(path);
    }

    private static <F> Classifier<Solution, String> createClassifierForName(String name, Approach<F> approach) {
        if (name.equals("closest-cluster")) {
            return new CompositeClassifier<>(
                    approach.extractor,
                    new NearestCluster<>(CommonUtils.metricFor(approach.metric, Wrapper::getFeatures)));
        } else if (name.equals("k-nearest-10")) {
            return new CompositeClassifier<>(
                    approach.extractor,
                    new KNearestNeighbors<>(10, CommonUtils.metricFor(approach.metric, Wrapper::getFeatures)));
        } else if (name.equals("k-nearest-5")) {
            return new CompositeClassifier<>(
                    approach.extractor,
                    new KNearestNeighbors<>(5, CommonUtils.metricFor(approach.metric, Wrapper::getFeatures)));
        } else if (name.equals("k-nearest-15")) {
            return new CompositeClassifier<>(
                    approach.extractor,
                    new KNearestNeighbors<>(15, CommonUtils.metricFor(approach.metric, Wrapper::getFeatures)));
        }
        return null;
    }

    public static double getAUC(List<Combiner.Test> points) {
        points.sort(Comparator.comparingDouble(Combiner.Test::getRecall)
                .thenComparingDouble(Combiner.Test::getPrecision)
                .thenComparingDouble(Combiner.Test::getThreshold));
        double auc = 0;
        double prevPrecision = 1;
        double prevRecall = 0;
        for (Combiner.Test point : points) {
            auc += (point.recall - prevRecall) * (point.precision + prevPrecision) / 2;
            prevPrecision = point.precision;
            prevRecall = point.recall;
        }
        auc += (1 - prevRecall) * prevPrecision / 2;
        return auc;
    }
}
