package org.ml_methods_group.classification;

import org.ml_methods_group.classification.classifiers.CompositeClassifier;
import org.ml_methods_group.classification.classifiers.KNearestNeighbors;
import org.ml_methods_group.common.*;
import org.ml_methods_group.common.changes.generation.BasicASTNormalizer;
import org.ml_methods_group.common.changes.generation.BasicChangeGenerator;
import org.ml_methods_group.common.changes.generation.ChangeGenerator;
import org.ml_methods_group.common.changes.generation.Changes;
import org.ml_methods_group.common.extractors.ChangesExtractor;
import org.ml_methods_group.common.metrics.functions.ChangeDistanceFunction;
import org.ml_methods_group.common.metrics.functions.FuzzyListDistanceFunction;
import org.ml_methods_group.common.metrics.functions.HeuristicChangesBasedDistanceFunction;
import org.ml_methods_group.common.metrics.selectors.ClosestPairSelector;
import org.ml_methods_group.common.serialization.MarkedSolutionsClusters;
import org.ml_methods_group.common.serialization.SolutionClassifier;
import org.ml_methods_group.common.serialization.SolutionsDataset;

import java.util.List;
import java.util.TreeSet;
import java.util.stream.Collectors;

import static org.ml_methods_group.common.Solution.Verdict.OK;

public class ClassificationUtils {
    public static <V> List<V> kNearest(V value, List<V> targets, int k, DistanceFunction<V> metric) {
        double bound = Double.POSITIVE_INFINITY;
        final TreeSet<Wrapper<Double, Integer>> heap = new TreeSet<>(Wrapper::compare);
        for (int i = 0; i < targets.size(); i++) {
            final double distance = metric.distance(value, targets.get(i), bound);
            if (heap.size() < k || distance < bound) {
                heap.add(new Wrapper<>(distance, i));
                if (heap.size() > k) {
                    heap.pollLast();
                }
                bound = heap.last().getFeatures();
            }
        }
        return heap.stream()
                .mapToInt(Wrapper::getMeta)
                .mapToObj(targets::get)
                .collect(Collectors.toList());
    }

    public static SolutionClassifier trainClassifier(MarkedSolutionsClusters clusters, SolutionsDataset dataset) {
        final List<Solution> correct = dataset.getValues(CommonUtils.check(Solution::getVerdict, OK::equals));

        final ChangeGenerator generator = new BasicChangeGenerator(new BasicASTNormalizer());
        final DistanceFunction<Solution> solutionsMetric = new HeuristicChangesBasedDistanceFunction(generator);
        final DistanceFunction<Changes> changesMetric = CommonUtils.metricFor(
                new FuzzyListDistanceFunction<>(new ChangeDistanceFunction(), ChangeDistanceFunction::getChangeClass),
                Changes::getChanges);

        final FeaturesExtractor<Solution, Changes> extractor = new ChangesExtractor(
                new BasicChangeGenerator(new BasicASTNormalizer()),
                new ClosestPairSelector<>(correct, solutionsMetric));

        final KNearestNeighbors<Wrapper<Changes, Solution>, String> knn = new KNearestNeighbors<>(
                10,
                CommonUtils.metricFor(changesMetric, Wrapper::getFeatures));
        final CompositeClassifier<Solution, Changes, String> adapter = new CompositeClassifier<>(extractor, knn);
        final SolutionClassifier classifier = new SolutionClassifier(adapter);
        classifier.train(clusters);
        return classifier;
    }
}
