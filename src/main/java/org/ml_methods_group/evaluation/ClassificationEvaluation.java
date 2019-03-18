package org.ml_methods_group.evaluation;

import org.ml_methods_group.cache.HashDatabase;
import org.ml_methods_group.classification.classifiers.CompositeClassifier;
import org.ml_methods_group.classification.classifiers.KNearestNeighbors;
import org.ml_methods_group.classification.classifiers.NearestCluster;
import org.ml_methods_group.common.*;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.common.serialization.MarkedSolutionsClusters;
import org.ml_methods_group.common.serialization.SolutionsDataset;
import org.ml_methods_group.evaluation.approaches.Approach;
import org.ml_methods_group.evaluation.approaches.VectorizationApproach;
import org.ml_methods_group.testing.BasicClassificationTester;
import org.ml_methods_group.testing.ClassificationTester;
import org.ml_methods_group.testing.ClassificationTestingResult;
import org.ml_methods_group.testing.validation.Validator;
import org.ml_methods_group.testing.validation.basic.PrecalculatedValidator;


import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import static org.ml_methods_group.common.Solution.Verdict.FAIL;
import static org.ml_methods_group.evaluation.ClusterizationEvaluation.createChangesExtractor;

public class ClassificationEvaluation {

//    public static String problem = "min_max";
    public static String problem = "double_equality";
//    public static String problem = "loggers";
//    public static String problem = "deserialization";

    public static String dataset = "validate";
    //    public static String classifier = "k-nearest";
    public static String[] classifiers = {"closest-cluster", "k-nearest-5", "k-nearest-10", "k-nearest-15"};

    public static void main(String[] args) throws Exception {
        SolutionsDataset train = SolutionsDataset.load(Paths.get(".cache", "datasets", problem, "train.tmp"));
        final List<Solution> incorrect = SolutionsDataset
                .load(Paths.get(".cache", "datasets", problem, dataset + ".tmp"))
                .getValues(x -> x.getVerdict() == FAIL);
        final Database database = new HashDatabase(EvaluationInfo.pathToCache);
        final FeaturesExtractor<Solution, Changes> changesExtractor = createChangesExtractor(train, database);
        System.out.println("Extractor created");
        long start = System.currentTimeMillis();

//        final Approach<BOWVector> approach = BOWApproach.getDefaultApproach(20000, train, changesExtractor);
//        final Approach<List<String>> approach = JaccardApproach.getDefaultApproach(changesExtractor);
//        final Approach<List<String>> approach = JaccardApproach.getExtendedApproach(changesExtractor);
//        final Approach<List<String>> approach = JaccardApproach.getFullApproach(changesExtractor);
//        final Approach<Changes> approach = FuzzyJaccardApproach.getDefaultApproach(changesExtractor);
//        final Approach<List<double[]>> approach = VectorizationApproach.getDefaultApproach(train, changesExtractor);
//        final Approach<double[]> approach = VectorizationApproach.getSumApproach(train, changesExtractor);
        final Approach<double[]> approach = VectorizationApproach.getMeanApproach(train, changesExtractor);
//        System.out.println("Approach initialized");

        final Validator<Solution, String> validator = loadValidator(problem);
        try (PrintWriter out = new PrintWriter(approach.name + ".csv")) {
            final Path path = Paths.get(".cache", "marked_clusters", problem);
            for (File approachFolder : path.toFile().listFiles()) {
                for (File file : approachFolder.listFiles()) {
                    for (String classifier : classifiers) {
                        final MarkedSolutionsClusters clusters = MarkedSolutionsClusters.load(file.toPath());
                        final String prefix = approachFolder.getName() + "," + getData(file.getName())
                                + "," + classifier + ",";
                        evaluate(createClassifierForName(classifier, approach), clusters, incorrect, validator,
                                prefix, out);
                    }
                }
                System.out.println(approachFolder.getName() + " clusters tested");
            }
        }
        System.out.println((System.currentTimeMillis() - start) / 1000);
    }

    private static String getData(String filename) {
        return filename.substring(0, filename.lastIndexOf('.')).replaceAll("_", ",");
    }

    private static Validator<Solution, String> loadValidator(String problem) throws IOException {
        final Path path = Paths.get(".cache", "validators", problem + ".pvd");
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

    private static <F> void evaluate(Classifier<Solution, String> classifier, MarkedSolutionsClusters clusters,
                                     List<Solution> test, Validator<Solution, String> validator,
                                     String prefix, PrintWriter out) {
        classifier.train(clusters);
        final ClassificationTester<Solution, String> tester = new BasicClassificationTester<>(test, validator);
        final ClassificationTestingResult result1 = tester.test(classifier);
        double[] t = {0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5};
        for (double tr : t) {
            out.print(prefix);
            out.print(tr + ",");
            out.print(result1.getCoverage(tr) + ",");
            out.print(result1.getPrecision(tr) + ",");
            out.println(result1.getAccuracy());
        }
    }
}
