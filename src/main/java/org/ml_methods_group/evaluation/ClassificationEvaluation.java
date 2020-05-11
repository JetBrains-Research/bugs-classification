package org.ml_methods_group.evaluation;

import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.cache.HashDatabase;
import org.ml_methods_group.common.*;
import org.ml_methods_group.common.ast.ASTUtils;
import org.ml_methods_group.common.ast.changes.BasicChangeGenerator;
import org.ml_methods_group.common.ast.changes.ChangeGenerator;
import org.ml_methods_group.common.ast.generation.ASTGenerator;
import org.ml_methods_group.common.ast.generation.CachedASTGenerator;
import org.ml_methods_group.common.ast.normalization.NamesASTNormalizer;
import org.ml_methods_group.common.extractors.ChangesExtractor;
import org.ml_methods_group.common.metrics.functions.HeuristicChangesBasedDistanceFunction;
import org.ml_methods_group.common.metrics.selectors.ClosestPairSelector;
import org.ml_methods_group.common.preparation.Unifier;
import org.ml_methods_group.common.preparation.basic.BasicUnifier;
import org.ml_methods_group.common.preparation.basic.MinValuePicker;
import org.ml_methods_group.evaluation.approaches.BOWApproach;
import org.ml_methods_group.evaluation.approaches.FuzzyJaccardApproach;
import org.ml_methods_group.evaluation.approaches.JaccardApproach;
import org.ml_methods_group.evaluation.approaches.VectorizationApproach;
import org.ml_methods_group.evaluation.approaches.classification.ClassificationApproach;
import org.ml_methods_group.evaluation.approaches.classification.ClassificationApproachTemplate;
import org.ml_methods_group.testing.BasicClassificationTester;
import org.ml_methods_group.testing.ClassificationTestingResult;
import org.ml_methods_group.testing.selectors.CacheOptionSelector;

import java.io.PrintWriter;
import java.nio.file.Path;
import java.util.*;

import static org.ml_methods_group.common.Solution.Verdict.FAIL;
import static org.ml_methods_group.common.Solution.Verdict.OK;
import static org.ml_methods_group.common.serialization.ProtobufSerializationUtils.*;

public class ClassificationEvaluation {

    public static int[] numClustersToMark = {5, 10, 20, 30, 40};
    public static String[] clusteringApproachesNames = {"def_jac", "ext_jac", "ful_jac", "fuz_jac", "BOW20000",
            "def_vec"};
    public static double[] clusteringHacThresholds = {0.1, 0.2, 0.3, 0.4, 0.5};
    public static double[] classificationThresholds =
            {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95};

    public static ClassificationApproachTemplate[] approaches = {
            new ClassificationApproachTemplate((dataset, extractor) ->
                    JaccardApproach.getDefaultApproach(extractor)),
            new ClassificationApproachTemplate((dataset, extractor) ->
                    JaccardApproach.getExtendedApproach(extractor)),
            new ClassificationApproachTemplate((dataset, extractor) ->
                    JaccardApproach.getFullApproach(extractor)),
            new ClassificationApproachTemplate((dataset, extractor) ->
                    FuzzyJaccardApproach.getDefaultApproach(extractor)),
            new ClassificationApproachTemplate((dataset, extractor) ->
                    BOWApproach.getDefaultApproach(20000, dataset, extractor)),
            new ClassificationApproachTemplate((dataset, extractor) ->
                    VectorizationApproach.getDefaultApproach(dataset, extractor)),
            new ClassificationApproachTemplate((dataset, extractor) ->
                    VectorizationApproach.getSumApproach(dataset, extractor)),
    };

    public static String[] classifiers = {"closest-cluster", "k-nearest-3",
            "k-nearest-5", "k-nearest-10", "k-nearest-15"
    };

    public static String[] problems = {
            "loggers",
            "deserialization",
            "reflection",
            "factorial"
    };

    public static void main(String[] args) throws Exception {
        try (Database database = new HashDatabase(EvaluationInfo.PATH_TO_CACHE)) {
            final ASTGenerator astGenerator = new CachedASTGenerator(new NamesASTNormalizer());
            final ChangeGenerator changeGenerator = new BasicChangeGenerator(astGenerator);
            final Unifier<Solution> unifier = new BasicUnifier<>(
                    CommonUtils.compose(astGenerator::buildTree, ITree::getHash)::apply,
                    CommonUtils.checkEquals(astGenerator::buildTree, ASTUtils::deepEquals),
                    new MinValuePicker<>(Comparator.comparingInt(Solution::getSolutionId)));
            for (String problem : problems) {
                final DistanceFunction<Solution> metric =
                        new HeuristicChangesBasedDistanceFunction(changeGenerator);
                final Path clusters = EvaluationInfo.PATH_TO_CLUSTERS.resolve(problem);
                final Path problemData = EvaluationInfo.PATH_TO_DATASET.resolve(problem);
                final Path validation = problemData.resolve("validation");
                final SolutionMarksHolder holder = loadSolutionMarksHolder(problemData.resolve("train_marks.tmp"));
                final Map<String, double[]> results = new HashMap<>();
                for (int k = 0; k < 10; k++) {
                    System.out.println("Step " + k);
                    final Dataset train = loadDataset(validation.resolve("step_" + k).resolve("train.tmp"));
                    final Dataset test = loadDataset(validation.resolve("step_" + k).resolve("validate.tmp"));
                    final List<Solution> correct = train.getValues(x -> x.getVerdict() == OK);
                    final OptionSelector<Solution, Solution> selector = new CacheOptionSelector<>(
                            new ClosestPairSelector<>(unifier.unify(correct), metric),
                            database,
                            Solution::getSolutionId,
                            Solution::getSolutionId);
                    final var generator = new ChangesExtractor(changeGenerator, selector);
                    final var tester = new BasicClassificationTester<>(test.getValues(x -> x.getVerdict() == FAIL),
                            (Solution s, String m) -> holder.getMarks(s).filter(x -> x.contains(m)).isPresent());
                    for (ClassificationApproachTemplate template : approaches) {
                        final ClassificationApproach approach = template.createApproach(train, generator);
                        System.out.println("    Start classification approach " + approach.getName());
                        for (String clusteringApproach : clusteringApproachesNames) {
                            for (double hacThreshold : clusteringHacThresholds) {
                                System.out.println("        Run clustering approach " + clusteringApproach + "_" + hacThreshold);
                                final Path testDir = clusters.resolve(clusteringApproach + "_" + hacThreshold)
                                        .resolve("step_" + k);
                                for (int numClusters : numClustersToMark) {
                                    final var data = loadMarkedClusters(testDir.resolve(numClusters + "_clusters.tmp"));
                                    for (String classifierName : classifiers) {
                                        final var classifier = approach.getClassifier(classifierName);
                                        classifier.train(data);
                                        final var result = getResults(tester.test(classifier));
                                        final String key = clusteringApproach + ',' +
                                                hacThreshold + ',' +
                                                numClusters + ',' +
                                                approach.getName() + ',' +
                                                classifierName;
                                        final double[] accumulator = results.computeIfAbsent(key,
                                                x -> new double[2 * classificationThresholds.length + 1]);
                                        for (int i = 0; i < accumulator.length; i++) {
                                            accumulator[i] += result[i] / 10;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                try (var out = new PrintWriter(EvaluationInfo.PATH_TO_RESULTS.resolve(problem + ".csv").toFile())) {
                    for (var entry : results.entrySet()) {
                        out.print(entry.getKey());
                        out.print(",");
                        Arrays.stream(entry.getValue()).forEach(x -> out.print(x + ","));
                        out.println(getAUC(entry.getValue()));
                    }
                }
            }
        }
    }

    private static double[] getResults(ClassificationTestingResult results) {
        final double[] buffer = new double[classificationThresholds.length * 2 + 1];
        for (int i = 0; i < classificationThresholds.length; i++) {
            buffer[2 * i] = results.getPrecision(classificationThresholds[i]);
            buffer[2 * i + 1] = results.getCoverage(classificationThresholds[i]);
        }
        buffer[buffer.length - 1] = results.getAccuracy();
        return buffer;
    }

    private static double getAUC(double[] data) {
        final List<Point> points = new ArrayList<>();
        for (int i = 0; i < classificationThresholds.length; i++) {
            points.add(new Point(data[2 * i], data[2 * i + 1]));
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
