package org.ml_methods_group.evaluation;

import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.cache.HashDatabase;
import org.ml_methods_group.common.*;
import org.ml_methods_group.common.ast.ASTUtils;
import org.ml_methods_group.common.ast.changes.BasicChangeGenerator;
import org.ml_methods_group.common.ast.changes.ChangeGenerator;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.common.ast.generation.ASTGenerator;
import org.ml_methods_group.common.ast.generation.CachedASTGenerator;
import org.ml_methods_group.common.ast.normalization.NamesASTNormalizer;
import org.ml_methods_group.common.extractors.ChangesExtractor;
import org.ml_methods_group.common.extractors.KNearestNeighborsChangesExtractor;
import org.ml_methods_group.common.metrics.functions.HeuristicChangesBasedDistanceFunction;
import org.ml_methods_group.common.metrics.selectors.ClosestPairSelector;
import org.ml_methods_group.common.metrics.selectors.KClosestPairsSelector;
import org.ml_methods_group.common.preparation.Unifier;
import org.ml_methods_group.common.preparation.basic.BasicUnifier;
import org.ml_methods_group.common.preparation.basic.MinValuePicker;
import org.ml_methods_group.evaluation.approaches.BOWApproach;
import org.ml_methods_group.evaluation.approaches.FuzzyJaccardApproach;
import org.ml_methods_group.evaluation.approaches.classification.ClassificationApproachTemplate;
import org.ml_methods_group.evaluation.approaches.clustering.ClusteringApproachTemplate;
import org.ml_methods_group.evaluation.preparation.TokenBasedDatasetsCreator;
import org.ml_methods_group.marking.markers.ManualClusterMarker;
import org.ml_methods_group.marking.markers.Marker;
import org.ml_methods_group.testing.BasicClassificationTester;
import org.ml_methods_group.testing.ClassificationTester;
import org.ml_methods_group.testing.ClassificationTestingResult;
import org.ml_methods_group.testing.selectors.CacheManyOptionsSelector;
import org.ml_methods_group.testing.selectors.CacheOptionSelector;

import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

import static org.ml_methods_group.common.Solution.Verdict.FAIL;
import static org.ml_methods_group.common.Solution.Verdict.OK;
import static org.ml_methods_group.common.serialization.ProtobufSerializationUtils.*;
import static org.ml_methods_group.evaluation.preparation.MarkedClustersSaver.createClusters;

public class PipelineEvaluation {

    public static final ClusteringApproachTemplate clusteringTemplate =
            new ClusteringApproachTemplate(((dataset, generator) ->
                    BOWApproach.getDefaultApproach(20000, dataset, generator)));

    public static final ClassificationApproachTemplate classificationTemplate =
            new ClassificationApproachTemplate(((dataset, generator) ->
                    FuzzyJaccardApproach.getDefaultApproach(generator)));

    public static CacheManyOptionsSelector<Solution, Solution> getCacheSelectorFromTemplate(
            KClosestPairsSelector<Solution> selector, Database database) throws Exception {
        return new CacheManyOptionsSelector<Solution, Solution>(
                selector,
                database,
                Solution::getSolutionId,
                Solution::getSolutionId,
                list -> list.stream()
                        .map(String::valueOf)
                        .collect(Collectors.joining(",")),
                string -> Arrays.stream(string.split(","))
                        .map(Integer::valueOf)
                        .collect(Collectors.toList())
        );
    }

    public static final List<String> problems = Arrays.asList("factorial", "loggers", "reflection", "deserialization");

    public static void main(String[] args) throws Exception {
        var problem = problems.get(1);
        final Path datasetPath = EvaluationInfo.PATH_TO_DATASET.resolve(problem);
        Path pathToTrain = datasetPath.resolve("train_tokens_dataset.txt");
        Path pathToTest = datasetPath.resolve("test_tokens_dataset.txt");

        System.out.println("Start clustering");
        createClusters(datasetPath);
        System.out.println("Clusters created and saved, starting creating datasets");
        runNewPipeline(datasetPath, pathToTrain, pathToTest);
        System.out.println("End creating datasets, start training classification model");
        runClassification(pathToTrain, pathToTest);
    }

    public static void runNewPipeline(Path globalDatasetPath, Path pathToTrain, Path pathToTest) throws Exception {
        try (final HashDatabase database = new HashDatabase(EvaluationInfo.PATH_TO_CACHE)) {
            final ASTGenerator astGenerator = new CachedASTGenerator(new NamesASTNormalizer());
            final ChangeGenerator changeGenerator = new BasicChangeGenerator(astGenerator);
            final Unifier<Solution> unifier = new BasicUnifier<>(
                    CommonUtils.compose(astGenerator::buildTree, ITree::getHash)::apply,
                    CommonUtils.checkEquals(astGenerator::buildTree, ASTUtils::deepEquals),
                    new MinValuePicker<>(Comparator.comparingInt(Solution::getSolutionId)));
            final DistanceFunction<Solution> metric = new HeuristicChangesBasedDistanceFunction(changeGenerator);

            final SolutionMarksHolder testHolder = loadSolutionMarksHolder(globalDatasetPath.resolve("test_marks.tmp"));
            final Dataset train = loadDataset(globalDatasetPath.resolve("train.tmp"));
            final Dataset test = loadDataset(globalDatasetPath.resolve("test.tmp"));
            final MarkedClusters<Solution, String> markedClusters = loadMarkedClusters(globalDatasetPath.resolve("clusters.tmp"));
            final List<Solution> correctFromTrain = train.getValues(x -> x.getVerdict() == OK);
            final List<Solution> incorrectFromTrain = train.getValues(x -> x.getVerdict() == FAIL);
            final List<Solution> incorrectFromTest = test.getValues(x -> x.getVerdict() == FAIL);


            final var oneNearestSelector = getCacheSelectorFromTemplate(
                    new KClosestPairsSelector<>(unifier.unify(correctFromTrain), metric, 1), database);
            final var threeNearestSelector = getCacheSelectorFromTemplate(
                    new KClosestPairsSelector<>(unifier.unify(correctFromTrain), metric, 3), database);
            final FeaturesExtractor<Solution, List<Changes>> generator =
                    new KNearestNeighborsChangesExtractor(changeGenerator, oneNearestSelector);
            final FeaturesExtractor<Solution, List<Changes>> threeNearestGenerator =
                    new KNearestNeighborsChangesExtractor(changeGenerator, threeNearestSelector);

            var datasetCreator = new TokenBasedDatasetsCreator(20000, train, threeNearestGenerator);
            var testMarksDictionary = new HashMap<Solution, List<String>>();
            for (var entry : testHolder) {
                testMarksDictionary.put(entry.getKey(), entry.getValue());
            }
            datasetCreator.createCodeChangesDataset(incorrectFromTest, generator, testMarksDictionary, pathToTest);
            var trainMarksDictionary = new HashMap<Solution, List<String>>();
            var flatMarks = markedClusters.getFlatMarks();
            for (var solution : incorrectFromTrain) {
                trainMarksDictionary.put(solution, Collections.singletonList(flatMarks.get(solution)));
            }
            datasetCreator.createCodeChangesDataset(incorrectFromTrain, threeNearestGenerator, trainMarksDictionary, pathToTrain);
        }
    }

    public static void runClassification(Path pathToTrain, Path pathToTest) throws Exception {
        String pythonBinaryPath = EvaluationInfo.PATH_TO_PYTHON_BINARY.toString();
        Path pythonScriptPath = EvaluationInfo.PATH_TO_PYTHON_SCRIPTS.resolve("classification.py");
        String[] command = {pythonBinaryPath, pythonScriptPath.toString(),
                "--train", pathToTrain.toString(), "--validate", pathToTest.toString()};
        ProcessBuilder pb = new ProcessBuilder(command);
        pb.redirectOutput(ProcessBuilder.Redirect.INHERIT);
        pb.redirectError(ProcessBuilder.Redirect.INHERIT);
        Process process = pb.start();
        try {
            process.waitFor();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } finally {
            process.destroy();
        }
    }

    public static void runOldPipeline(String problem) throws Exception {
        try (final HashDatabase database = new HashDatabase(EvaluationInfo.PATH_TO_CACHE)) {
            final ASTGenerator astGenerator = new CachedASTGenerator(new NamesASTNormalizer());
            final ChangeGenerator changeGenerator = new BasicChangeGenerator(astGenerator);
            final Unifier<Solution> unifier = new BasicUnifier<>(
                    CommonUtils.compose(astGenerator::buildTree, ITree::getHash)::apply,
                    CommonUtils.checkEquals(astGenerator::buildTree, ASTUtils::deepEquals),
                    new MinValuePicker<>(Comparator.comparingInt(Solution::getSolutionId)));
            System.out.println("Start problem: " + problem);
            final DistanceFunction<Solution> metric =
                    new HeuristicChangesBasedDistanceFunction(changeGenerator);
            final Path datasetPath = EvaluationInfo.PATH_TO_DATASET.resolve(problem);
            final SolutionMarksHolder testHolder = loadSolutionMarksHolder(datasetPath.resolve("test_marks.tmp"));
            final Dataset dataset = loadDataset(datasetPath.resolve("train.tmp"));
            final Dataset test = loadDataset(datasetPath.resolve("test.tmp"));
            final ClassificationTester<Solution, String> tester = new BasicClassificationTester<>(
                    test.getValues(x -> x.getVerdict() == FAIL),
                    (solution, mark) -> testHolder.getMarks(solution).filter(x -> x.contains(mark)).isPresent());
            final List<Solution> correct = dataset.getValues(x -> x.getVerdict() == OK);
            final List<Solution> incorrect = dataset.getValues(x -> x.getVerdict() == FAIL);
            final OptionSelector<Solution, Solution> selector = new CacheOptionSelector<>(
                    new ClosestPairSelector<>(unifier.unify(correct), metric),
                    database,
                    Solution::getSolutionId,
                    Solution::getSolutionId);
            final FeaturesExtractor<Solution, Changes> generator = new ChangesExtractor(changeGenerator, selector);
            final Clusterer<Solution> clusterer = clusteringTemplate.createApproach(dataset, generator)
                    .getClusterer(0.3);
            final Clusters<Solution> clusters = clusterer.buildClusters(incorrect);
            final Classifier<Solution, String> classifier = classificationTemplate.createApproach(dataset, generator)
                    .getClassifier("k-nearest-15");
            classifier.train(markClusters(clusters));
            final ClassificationTestingResult result = tester.test(classifier);
            System.out.println("AUC: " + getAUC(result));
            System.out.println("(Recall, Precision)");
            for (double t = 0.05; t < 1; t += 0.05) {
                System.out.println("(" + result.getCoverage(t) + ", " + result.getPrecision(t) + ")");
            }
        }
    }

    private static MarkedClusters<Solution, String> markClusters(Clusters<Solution> clusters) {
        final List<Cluster<Solution>> list = clusters.getClusters().stream()
                .sorted(Comparator.<Cluster<Solution>>comparingInt(Cluster::size).reversed())
                .limit(40)
                .collect(Collectors.toList());
        final Marker<Cluster<Solution>, String> marker =
                new ManualClusterMarker(5);
        final Map<Cluster<Solution>, String> marks = new HashMap<>();
        for (Cluster<Solution> cluster : list) {
            final String mark = marker.mark(cluster);
            if (mark != null) {
                marks.put(cluster, mark);
            }
        }
        return new MarkedClusters<>(marks);
    }

    private static double getAUC(ClassificationTestingResult results) {
        final List<Point> points = new ArrayList<>();
        for (double t = 0.05; t < 1; t += 0.05) {
            points.add(new Point(results.getPrecision(t), results.getCoverage(t)));
        }
        Collections.sort(points);
        double auc = 0;
        double prevPrecision = 1;
        double prevRecall = 0;
        for (Point point : points) {
            auc += (point.recall - prevRecall) * (point.precision + prevPrecision) / 2;
            prevPrecision = point.precision;
            prevRecall = point.recall;
        }
        auc += (1 - prevRecall) * prevPrecision / 2;
        return auc;
    }

    private static class Point implements Comparable<Point> {
        private final double precision;
        private final double recall;

        private Point(double precision, double recall) {
            this.precision = precision;
            this.recall = recall;
        }

        @Override
        public int compareTo(Point o) {
            if (recall != o.recall) {
                return Double.compare(recall, o.recall);
            }
            return Double.compare(o.precision, precision);
        }
    }
}
