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
import org.ml_methods_group.common.preparation.Unifier;
import org.ml_methods_group.common.preparation.basic.BasicUnifier;
import org.ml_methods_group.common.preparation.basic.MinValuePicker;
import org.ml_methods_group.evaluation.EvaluationInfo;
import org.ml_methods_group.evaluation.approaches.BOWApproach;
import org.ml_methods_group.evaluation.approaches.clustering.ClusteringApproachTemplate;
import org.ml_methods_group.testing.selectors.CacheOptionSelector;

import java.nio.file.Path;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

import static org.ml_methods_group.common.Solution.Verdict.FAIL;
import static org.ml_methods_group.common.Solution.Verdict.OK;
import static org.ml_methods_group.common.serialization.ProtobufSerializationUtils.*;

public class MarkedClustersSaver {

    public static final ClusteringApproachTemplate clusteringTemplate =
            new ClusteringApproachTemplate(((dataset, generator) ->
                    BOWApproach.getDefaultApproach(20000, dataset, generator)));

    public static void createClusters(Path datasetPath) throws Exception {
        try (final HashDatabase database = new HashDatabase(EvaluationInfo.PATH_TO_CACHE)) {
            final ASTGenerator astGenerator = new CachedASTGenerator(new NamesASTNormalizer());
            final ChangeGenerator changeGenerator = new BasicChangeGenerator(astGenerator);
            final Unifier<Solution> unifier = new BasicUnifier<>(
                    CommonUtils.compose(astGenerator::buildTree, ITree::getHash)::apply,
                    CommonUtils.checkEquals(astGenerator::buildTree, ASTUtils::deepEquals),
                    new MinValuePicker<>(Comparator.comparingInt(Solution::getSolutionId)));
            final DistanceFunction<Solution> metric = new HeuristicChangesBasedDistanceFunction(changeGenerator);

            // Collect data
            final SolutionMarksHolder trainHolder = loadSolutionMarksHolder(datasetPath.resolve("extended.tmp"));
            final SolutionMarksHolder testHolder = loadSolutionMarksHolder(datasetPath.resolve("test_marks.tmp"));
            final Dataset train = loadDataset(datasetPath.resolve("train.tmp"));
            final Dataset test = loadDataset(datasetPath.resolve("test.tmp"));
            final List<Solution> correct = train.getValues(x -> x.getVerdict() == OK);
            final List<Solution> incorrect = train.getValues(x -> x.getVerdict() == FAIL);

            // Create clusters
            final OptionSelector<Solution, Solution> selector = new CacheOptionSelector<>(
                    new ClosestPairSelector<>(unifier.unify(correct), metric),
                    database,
                    Solution::getSolutionId,
                    Solution::getSolutionId);
            final FeaturesExtractor<Solution, Changes> generator = new ChangesExtractor(changeGenerator, selector);
            final Clusterer<Solution> clusterer = clusteringTemplate.createApproach(train, generator)
                    .getClusterer(0.3);
            final Clusters<Solution> clusters = clusterer.buildClusters(incorrect);

            // Add pre-calculated marks and store marked clusters
            final List<Cluster<Solution>> topClusters = clusters.getClusters().stream()
                    .sorted(Comparator.<Cluster<Solution>>comparingInt(Cluster::size).reversed())
                    .limit(51)
                    .collect(Collectors.toList());
            final Map<Cluster<Solution>, String> marks = new HashMap<>();
            for (Cluster<Solution> cluster : topClusters) {
                Map<String, Long> counters = cluster.getElements()
                        .stream()
                        .map(trainHolder::getMarks)
                        .map(list -> list.orElse(new ArrayList<String>()))
                        .flatMap(List::stream)
                        .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
                long maxCounter = Long.MIN_VALUE;
                long totalAmount = 0;
                String mark = "";
                for (Map.Entry<String, Long> it : counters.entrySet()) {
                    if (it.getValue() > maxCounter) {
                        maxCounter = it.getValue();
                        mark = it.getKey();
                    }
                    totalAmount += it.getValue();
                }
                if ((float) (maxCounter) / (float) (totalAmount) > 0.6)
                    marks.put(cluster, mark);
                marks.putIfAbsent(cluster, "");
            }
            final MarkedClusters<Solution, String> markedClusters = new MarkedClusters<>(marks);
            storeMarkedClusters(markedClusters, datasetPath.resolve("clusters.tmp"));

            // Print additional info
            System.out.println("clusters: " + clusters.getClusters().size());
            System.out.println("train_marks: " + trainHolder.size());
            System.out.println("test_marks: " + testHolder.size());
            System.out.println("train: " + train.getValues().size());
            System.out.println("test: " + test.getValues().size());
        }
    }
}
