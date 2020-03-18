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
import org.ml_methods_group.common.metrics.functions.HeuristicChangesBasedDistanceFunction;
import org.ml_methods_group.common.metrics.selectors.ClosestPairSelector;
import org.ml_methods_group.common.preparation.Unifier;
import org.ml_methods_group.common.preparation.basic.BasicUnifier;
import org.ml_methods_group.common.preparation.basic.MinValuePicker;
import org.ml_methods_group.common.serialization.ProtobufSerializationUtils;
import org.ml_methods_group.evaluation.approaches.BOWApproach;
import org.ml_methods_group.evaluation.approaches.FuzzyJaccardApproach;
import org.ml_methods_group.evaluation.approaches.JaccardApproach;
import org.ml_methods_group.evaluation.approaches.VectorizationApproach;
import org.ml_methods_group.evaluation.approaches.clustering.ClusteringApproach;
import org.ml_methods_group.evaluation.approaches.clustering.ClusteringApproachTemplate;
import org.ml_methods_group.marking.markers.Marker;
import org.ml_methods_group.testing.markers.MarksHolderBasedMarker;
import org.ml_methods_group.testing.markers.SolutionMarksHolderExpander;
import org.ml_methods_group.testing.selectors.CacheOptionSelector;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

import static org.ml_methods_group.common.Solution.Verdict.FAIL;
import static org.ml_methods_group.common.Solution.Verdict.OK;
import static org.ml_methods_group.common.serialization.ProtobufSerializationUtils.*;

public class ClusterizationEvaluation {

    public static int[] numClustersToMark = {5, 10, 20, 30, 40};

    public static double[] hacThresholds = {0.1, 0.2, 0.3, 0.4, 0.5};

    public static ClusteringApproachTemplate[] approaches = {
            new ClusteringApproachTemplate((dataset, extractor) ->
                    JaccardApproach.getDefaultApproach(extractor)),
            new ClusteringApproachTemplate((dataset, extractor) ->
                    JaccardApproach.getExtendedApproach(extractor)),
            new ClusteringApproachTemplate((dataset, extractor) ->
                    JaccardApproach.getFullApproach(extractor)),
            new ClusteringApproachTemplate((dataset, extractor) ->
                    FuzzyJaccardApproach.getDefaultApproach(extractor)),
            new ClusteringApproachTemplate((dataset, extractor) ->
                    BOWApproach.getDefaultApproach(20000, dataset, extractor)),
            new ClusteringApproachTemplate((dataset, extractor) ->
                    VectorizationApproach.getDefaultApproach(dataset, extractor)),
    };

    public static String[] problems = {
            "loggers",
            "deserialization",
            "reflection",
            "factorial",
    };


    public static void main(String[] args) throws Exception {
        try (final HashDatabase database = new HashDatabase(EvaluationInfo.PATH_TO_CACHE);
             Scanner input = new Scanner(System.in)) {
            final ASTGenerator astGenerator = new CachedASTGenerator(new NamesASTNormalizer());
            final ChangeGenerator changeGenerator = new BasicChangeGenerator(astGenerator);
            final Unifier<Solution> unifier = new BasicUnifier<>(
                    CommonUtils.compose(astGenerator::buildTree, ITree::getHash)::apply,
                    CommonUtils.checkEquals(astGenerator::buildTree, ASTUtils::deepEquals),
                    new MinValuePicker<>(Comparator.comparingInt(Solution::getSolutionId)));
            for (String problem : problems) {
                System.out.println("Start problem: " + problem);
                final DistanceFunction<Solution> metric =
                        new HeuristicChangesBasedDistanceFunction(changeGenerator);
                final Path dataset = EvaluationInfo.PATH_TO_DATASET.resolve(problem);
                final Path results = EvaluationInfo.PATH_TO_CLUSTERS.resolve(problem);
                final Path validation = dataset.resolve("validation");
                final SolutionMarksHolder holder = loadExtendedHolder(dataset);
                for (int i = 0; i < 10; i++) {
                    System.out.println("    Start step: " + i);
                    final Path dataPath = validation.resolve("step_" + i);
                    final Dataset train = loadDataset(dataPath.resolve("train.tmp"));
                    final List<Solution> correct = train.getValues(x -> x.getVerdict() == OK);
                    final List<Solution> incorrect = train.getValues(x -> x.getVerdict() == FAIL);
                    final OptionSelector<Solution, Solution> selector = new CacheOptionSelector<>(
                            new ClosestPairSelector<>(unifier.unify(correct), metric),
                            database,
                            Solution::getSolutionId,
                            Solution::getSolutionId);
                    final FeaturesExtractor<Solution, Changes> generator = new ChangesExtractor(changeGenerator, selector);
                    final SolutionMarksHolderExpander expander =
                            new SolutionMarksHolderExpander(selector, changeGenerator, input, System.out);
                    for (ClusteringApproachTemplate template : approaches) {
                        final ClusteringApproach approach = template.createApproach(train, generator);
                        System.out.println("        Start approach: " + approach.getName());
                        for (double threshold : hacThresholds) {
                            System.out.println("            Start threshold: " + threshold);
                            final Clusterer<Solution> clusterer = approach.getClusterer(threshold);
                            final Clusters<Solution> clusters = clusterer.buildClusters(incorrect);
                            for (var result : markClusters(clusters, holder, expander).entrySet()) {
                                final Path path = results.resolve(approach.getName() + "_" + threshold)
                                        .resolve("step_" + i)
                                        .resolve(result.getKey() + "_clusters.tmp");
                                storeMarkedClusters(result.getValue(), path);
                            }
                        }
                    }
                }
                ProtobufSerializationUtils.storeSolutionMarksHolder(holder, dataset.resolve("extended.tmp"));
            }
        }
    }

    private static SolutionMarksHolder loadExtendedHolder(Path path) throws IOException {
        try {
            return loadSolutionMarksHolder(path.resolve("extended.tmp"));
        } catch (Exception e) {
            return loadSolutionMarksHolder(path.resolve("train_marks.tmp"));
        }
    }

    private static Map<Integer, MarkedClusters<Solution, String>> markClusters(Clusters<Solution> clusters,
                                                                               SolutionMarksHolder holder,
                                                                               SolutionMarksHolderExpander expander) {
        final List<Cluster<Solution>> list = clusters.getClusters().stream()
                .sorted(Comparator.<Cluster<Solution>>comparingInt(Cluster::size).reversed())
                .limit(numClustersToMark[numClustersToMark.length - 1])
                .collect(Collectors.toList());
        final Map<Integer, MarkedClusters<Solution, String>> result = new HashMap<>();
        final Marker<Cluster<Solution>, String> marker =
                new MarksHolderBasedMarker(holder, 0.8, 3, expander);
        final Map<Cluster<Solution>, String> marks = new HashMap<>();
        int pointer = 0;
        for (int i = 0; i < list.size(); i++) {
            final String mark = marker.mark(list.get(i));
            if (mark != null) {
                marks.put(list.get(i), mark);
            }
            if (i + 1 == numClustersToMark[pointer]) {
                result.put(numClustersToMark[pointer], new MarkedClusters<>(marks));
                pointer++;
            }
        }
        return result;
    }
}
