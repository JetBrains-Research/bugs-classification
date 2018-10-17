package org.ml_methods_group.clustering;

import org.ml_methods_group.clustering.clusterers.CompositeClusterer;
import org.ml_methods_group.clustering.clusterers.HAC;
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
import org.ml_methods_group.common.serialization.SolutionsClusterer;
import org.ml_methods_group.common.serialization.SolutionsClusters;
import org.ml_methods_group.common.serialization.SolutionsDataset;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import static org.ml_methods_group.common.Solution.Verdict.FAIL;
import static org.ml_methods_group.common.Solution.Verdict.OK;

public class ClusterizationUtils {
    public static SolutionsClusters buildClusters(SolutionsDataset dataset) {
        final List<Solution> correct = dataset.getValues(CommonUtils.check(Solution::getVerdict, OK::equals));
        final List<Solution> incorrect = dataset.getValues(CommonUtils.check(Solution::getVerdict, FAIL::equals));
        final Set<Integer> correctId = correct.stream()
                .map(Solution::getSessionId)
                .collect(Collectors.toSet());
        // todo may be shouldn't remove them
        incorrect.removeIf(CommonUtils.checkNot(Solution::getSessionId, correctId::contains));

        final ChangeGenerator generator = new BasicChangeGenerator(new BasicASTNormalizer());
        final DistanceFunction<Solution> solutionsMetric = new HeuristicChangesBasedDistanceFunction(generator);
        final DistanceFunction<Changes> changesMetric = CommonUtils.metricFor(
                new FuzzyListDistanceFunction<>(new ChangeDistanceFunction(), ChangeDistanceFunction::getChangeClass),
                Changes::getChanges);

        final FeaturesExtractor<Solution, Changes> extractor = new ChangesExtractor(
                new BasicChangeGenerator(new BasicASTNormalizer()),
                new ClosestPairSelector<>(correct, solutionsMetric));
        final HAC<Wrapper<Changes, Solution>> hac = new HAC<>(
                0.5,
                10,
                CommonUtils.metricFor(changesMetric, Wrapper::getFeatures));
        final CompositeClusterer<Solution, Changes> adapter = new CompositeClusterer<>(extractor, hac);
        final SolutionsClusterer clusterer = new SolutionsClusterer(adapter);
        return clusterer.buildClusters(incorrect);
    }
}
