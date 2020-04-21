package org.ml_methods_group.evaluation.preparation;

import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.cache.HashDatabase;
import org.ml_methods_group.clustering.clusterers.ClusterSizeLimitedHAC;
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
import org.ml_methods_group.evaluation.approaches.clustering.ClusteringApproach;
import org.ml_methods_group.evaluation.approaches.clustering.ClusteringApproachTemplate;
import org.ml_methods_group.marking.markers.ManualClusterMarker;
import org.ml_methods_group.marking.markers.Marker;
import org.ml_methods_group.testing.selectors.CacheOptionSelector;

import java.nio.file.Path;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collector;
import java.util.stream.Collectors;

import static org.ml_methods_group.common.Solution.Verdict.FAIL;
import static org.ml_methods_group.common.Solution.Verdict.OK;
import static org.ml_methods_group.common.serialization.ProtobufSerializationUtils.*;

public class ClustersCreator {

    private final HashDatabase database;
    private final Unifier<Solution> unifier;
    private final ChangeGenerator changeGenerator;
    private final DistanceFunction<Solution> metric;

    public static final ClusteringApproachTemplate clusteringTemplate =
            new ClusteringApproachTemplate(((dataset, generator) ->
                    BOWApproach.getDefaultApproach(20000, dataset, generator)));

    public static String[] problems = {
            "loggers",
            "deserialization",
            "reflection",
            "factorial",
    };

    public static void main(String[] argv) throws Exception {
        ClustersCreator creator = new ClustersCreator();
        final Dataset full = loadDataset(EvaluationInfo.PATH_TO_DATASET.resolve("dataset1.tmp"));
        creator.createGlobalClusters(full.getValues().stream()
                .map(Solution::getProblemId)
                .collect(Collectors.toSet())
                .stream()
                .map(Object::toString).toArray(String[]::new));

//        final List<Cluster<Solution>> clusters =
//                loadSolutionClusters(EvaluationInfo.PATH_TO_CLUSTERS.resolve("global_clusters.tmp"))
//                        .getClusters().stream()
//                        .sorted(Comparator.<Cluster<Solution>>comparingInt(Cluster::size).reversed())
//                        .limit(50)
//                        .collect(Collectors.toList());
//        SolutionMarksHolder localMarksHolder = new SolutionMarksHolder();
//        for (String problem : problems) {
//            final SolutionMarksHolder trainHolder =
//                    loadSolutionMarksHolder(EvaluationInfo.PATH_TO_DATASET.resolve(problem).resolve("extended.tmp"));
//            trainHolder.forEach(e -> {
//                Solution solution = e.getKey();
//                e.getValue().forEach(mark -> localMarksHolder.addMark(solution, mark));
//            });
//
//        }
//        storeMarkedClusters(
//                markClusters(clusters, localMarksHolder),
//                EvaluationInfo.PATH_TO_CLUSTERS.resolve("marked_global_clusters.tmp")
//        );
    }

    private static MarkedClusters<Solution, String> markClusters(List<Cluster<Solution>> clusters,
                                                                 SolutionMarksHolder localMarksHolder) {
        final Marker<Cluster<Solution>, String> marker =
                new ManualClusterMarker(10);
        final Map<Cluster<Solution>, String> marks = new HashMap<>();
        for (Cluster<Solution> cluster : clusters) {
            Map<String, Long> counters = cluster.getElements()
                    .stream()
                    .map(localMarksHolder::getMarks)
                    .map(list -> list.orElse(new ArrayList<String>()))
                    .flatMap(List::stream)
                    .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
            System.out.println("Local marks for current cluster: ");
            counters.forEach((mark, cnt) -> System.out.println("    " + mark + " - " + cnt));
            System.out.println();
            final String mark = marker.mark(cluster);
            if (mark != null) {
                marks.put(cluster, mark);
            }
        }
        return new MarkedClusters<>(marks);
    }


    public ClustersCreator() {
        this.database = new HashDatabase(EvaluationInfo.PATH_TO_CACHE);
        final ASTGenerator astGenerator = new CachedASTGenerator(new NamesASTNormalizer());
        this.changeGenerator = new BasicChangeGenerator(astGenerator);
        this.unifier = new BasicUnifier<>(
                CommonUtils.compose(astGenerator::buildTree, ITree::getHash)::apply,
                CommonUtils.checkEquals(astGenerator::buildTree, ASTUtils::deepEquals),
                new MinValuePicker<>(Comparator.comparingInt(Solution::getSolutionId)));
        this.metric = new HeuristicChangesBasedDistanceFunction(changeGenerator);
    }


    public void createGlobalClusters(String[] problems) throws Exception {
        final var generatorByDataset = new HashMap<Dataset, FeaturesExtractor<Solution, Changes>>();
        final var incorrect = new ArrayList<Solution>();

        // Merge solutions and generators from all problems
        for (String problem : problems) {
            final Dataset train = loadDataset(EvaluationInfo.PATH_TO_DATASET.resolve(problem).resolve("train.tmp"));
            final List<Solution> correct = train.getValues(x -> x.getVerdict() == OK);
            final var selector = new CacheOptionSelector<>(
                    new ClosestPairSelector<>(unifier.unify(correct), metric),
                    database,
                    Solution::getSolutionId,
                    Solution::getSolutionId
            );
            generatorByDataset.put(train, new ChangesExtractor(changeGenerator, selector));
            incorrect.addAll(train.getValues(x -> x.getVerdict() == FAIL));
        }
        System.out.println(incorrect.size());

        // Create global clusters based on edit scripts to nearest solution from the same problem
        final ClusteringApproach approach = new ClusteringApproach(
                "SPARSE_BOW_ALL_PROBLEMS",
                BOWApproach.getManyProblemsBasedApproach(generatorByDataset)
        );
        final Clusterer<Solution> clusterer = approach.getClusterer(0.3);
        Clusters<Solution> globalClusters = clusterer.buildClusters(incorrect);
        storeSolutionClusters(globalClusters, EvaluationInfo.PATH_TO_CLUSTERS.resolve("global_clusters.tmp"));

        System.out.println(globalClusters.getClusters().size());
        globalClusters.getClusters().stream()
                .sorted(Comparator.<Cluster<Solution>>comparingInt(Cluster::size).reversed())
                .collect(Collectors.toList())
                .forEach(x -> System.out.print(x.size() + " "));
    }


    public void createMarkedClusters(Path pathToDataset) throws Exception {
        // Collect data
        final SolutionMarksHolder trainHolder = loadSolutionMarksHolder(pathToDataset.resolve("extended.tmp"));
        final SolutionMarksHolder testHolder = loadSolutionMarksHolder(pathToDataset.resolve("test_marks.tmp"));
        final Dataset train = loadDataset(pathToDataset.resolve("train.tmp"));
        final Dataset test = loadDataset(pathToDataset.resolve("test.tmp"));
        final List<Solution> correct = train.getValues(x -> x.getVerdict() == OK);
        final List<Solution> incorrect = train.getValues(x -> x.getVerdict() == FAIL);

        // Create clusters based on edit scripts
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
        storeMarkedClusters(markedClusters, pathToDataset.resolve("clusters.tmp"));

        System.out.println("clusters: " + clusters.getClusters().size());
        System.out.println("train_marks: " + trainHolder.size());
        System.out.println("test_marks: " + testHolder.size());
        System.out.println("train: " + train.getValues().size());
        System.out.println("test: " + test.getValues().size());
    }


    public void createCorrectSolutionsClusters(String[] problems) throws Exception {
        for (String problem : problems) {
            Path pathToDataset = EvaluationInfo.PATH_TO_DATASET.resolve(problem);
            final Dataset train = loadDataset(pathToDataset.resolve("train.tmp"));
            final Dataset test = loadDataset(pathToDataset.resolve("test.tmp"));
            final List<Solution> unifiedCorrect = unifier.unify(train.getValues(x -> x.getVerdict() == OK));

            // Create clusters based on correct solutions
            int threshold = 100;
            int maxClusterSize = (int) Math.round(Math.sqrt(unifiedCorrect.size()));
            final Clusterer<Solution> clusterer = new ClusterSizeLimitedHAC<>(threshold, maxClusterSize, metric);
            final Clusters<Solution> clusters = clusterer.buildClusters(unifiedCorrect);
            storeSolutionClusters(clusters, pathToDataset.resolve("sqrt-clusters-" + threshold + ".tmp"));

            System.out.println(unifiedCorrect.size());
            clusters.getClusters().stream()
                    .sorted(Comparator.<Cluster<Solution>>comparingInt(Cluster::size).reversed())
                    .collect(Collectors.toList())
                    .forEach(x -> System.out.print(x.size() + " "));
        }
    }

}
