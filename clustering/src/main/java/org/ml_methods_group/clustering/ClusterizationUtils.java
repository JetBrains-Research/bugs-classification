package org.ml_methods_group.clustering;

import org.ml_methods_group.common.CommonUtils;
import org.ml_methods_group.common.Solution;
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
//
//        final ChangeGenerator generator = new BasicChangeGenerator(new CachedASTGenerator(new BasicASTNormalizer()));
//        final DistanceFunction<Solution> solutionsMetric = new HeuristicChangesBasedDistanceFunction(generator);
//        final DistanceFunction<Changes> changesMetric = CommonUtils.metricFor(
//                new FuzzyJaccardDistanceFunction<>(new ChangeSimilarityMetric(), ChangeSimilarityMetric::getChangeClass),
//                Changes::getChanges);
//
//        final FeaturesExtractor<Solution, Changes> extractor = new ChangesExtractor(
//                new BasicChangeGenerator(new BasicASTNormalizer()),
//                new ClosestPairSelector<>(correct, solutionsMetric));
//        final HAC<Wrapper<Changes, Solution>> hac = new HAC<>(
//                0.5,
//                10,
//                CommonUtils.metricFor(changesMetric, Wrapper::getFeatures));
//        final CompositeClusterer<Solution, Changes> adapter = new CompositeClusterer<>(extractor, hac);
//        final SolutionsClusterer clusterer = new SolutionsClusterer(adapter);
//        return clusterer.buildClusters(incorrect);
        return null;
    }
}
