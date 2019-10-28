package org.ml_methods_group.evaluation.preparation;

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
import org.ml_methods_group.common.metrics.functions.HeuristicChangesBasedDistanceFunction;
import org.ml_methods_group.common.metrics.selectors.ClosestPairSelector;
import org.ml_methods_group.common.metrics.selectors.KClosestPairsSelector;
import org.ml_methods_group.common.preparation.Unifier;
import org.ml_methods_group.common.preparation.basic.BasicUnifier;
import org.ml_methods_group.common.preparation.basic.MinValuePicker;
import org.ml_methods_group.evaluation.EvaluationInfo;
import org.ml_methods_group.evaluation.approaches.Approach;
import org.ml_methods_group.evaluation.approaches.BOWApproach;
import org.ml_methods_group.evaluation.approaches.FuzzyJaccardApproach;
import org.ml_methods_group.evaluation.approaches.classification.ClassificationApproachTemplate;
import org.ml_methods_group.evaluation.approaches.clustering.ClusteringApproach;
import org.ml_methods_group.evaluation.approaches.clustering.ClusteringApproachTemplate;
import org.ml_methods_group.marking.markers.ManualClusterMarker;
import org.ml_methods_group.marking.markers.Marker;
import org.ml_methods_group.testing.BasicClassificationTester;
import org.ml_methods_group.testing.ClassificationTester;
import org.ml_methods_group.testing.ClassificationTestingResult;
import org.ml_methods_group.testing.selectors.CacheOptionSelector;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.sql.SQLOutput;
import java.util.*;
import java.util.stream.Collectors;

import static org.ml_methods_group.common.Solution.Verdict.FAIL;
import static org.ml_methods_group.common.Solution.Verdict.OK;
import static org.ml_methods_group.common.serialization.ProtobufSerializationUtils.*;

public class CSVCreator {

    public static final ClusteringApproachTemplate clusteringTemplate =
            new ClusteringApproachTemplate(((dataset, generator) ->
                    BOWApproach.getDefaultApproach(20000, dataset, generator)));

    public static final ClassificationApproachTemplate classificationTemplate =
            new ClassificationApproachTemplate(((dataset, generator) ->
                    FuzzyJaccardApproach.getDefaultApproach(generator)));

    public static final String problem = "deserialization";

    public static void main(String[] args) throws Exception {
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
            final Dataset train = loadDataset(datasetPath.resolve("train.tmp"));
            final Dataset test = loadDataset(datasetPath.resolve("test.tmp"));
            final MarkedClusters<Solution, String> markedClusters
                    = loadMarkedClusters(datasetPath.resolve("clusters.tmp"));

            final ClassificationTester<Solution, String> tester = new BasicClassificationTester<>(
                    test.getValues(x -> x.getVerdict() == FAIL),
                    (solution, mark) -> testHolder.getMarks(solution).filter(x -> x.contains(mark)).isPresent());

            final List<Solution> correctFromTrain = train.getValues(x -> x.getVerdict() == OK);
            final List<Solution> incorrectFromTest = test.getValues(x -> x.getVerdict() == FAIL);
            final List<Solution> incorrectFromTrain = train.getValues(x -> x.getVerdict() == FAIL);
            final OptionSelector<Solution, Solution> selector = new CacheOptionSelector<>(
                    new ClosestPairSelector<>(unifier.unify(correctFromTrain), metric),
                    database,
                    Solution::getSolutionId,
                    Solution::getSolutionId);
            final FeaturesExtractor<Solution, Changes> generator = new ChangesExtractor(changeGenerator, selector);

            var marksDictionary = new HashMap<Solution, List<String>>();

            for (var entry : testHolder) {
                marksDictionary.put(entry.getKey(), entry.getValue());
            }

            /*var flatMarks = markedClusters.getFlatMarks();
            for (var solution : incorrectFromTrain) {
                marksDictionary.put(solution, Collections.singletonList(flatMarks.get(solution)));
            }*/

            BOWApproach.createCSV(5000, incorrectFromTest, generator, marksDictionary,
                    datasetPath.resolve("test_dataset.csv"));

            final Classifier<Solution, String> classifier = classificationTemplate.createApproach(train, generator)
                    .getClassifier("k-nearest-15");
            classifier.train(markedClusters);
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
