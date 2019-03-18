package org.ml_methods_group.evaluation;

import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.cache.HashDatabase;
import org.ml_methods_group.clustering.clusterers.CompositeClusterer;
import org.ml_methods_group.clustering.clusterers.HAC;
import org.ml_methods_group.common.*;
import org.ml_methods_group.common.ast.ASTUtils;
import org.ml_methods_group.common.ast.changes.BasicChangeGenerator;
import org.ml_methods_group.common.ast.changes.ChangeGenerator;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.common.ast.generation.ASTGenerator;
import org.ml_methods_group.common.ast.generation.CachedASTGenerator;
import org.ml_methods_group.common.ast.normalization.NamesASTNormalizer;
import org.ml_methods_group.common.extractors.BOWExtractor.BOWVector;
import org.ml_methods_group.common.extractors.ChangesExtractor;
import org.ml_methods_group.common.metrics.functions.HeuristicChangesBasedDistanceFunction;
import org.ml_methods_group.common.metrics.selectors.ClosestPairSelector;
import org.ml_methods_group.common.preparation.Unifier;
import org.ml_methods_group.common.preparation.basic.BasicUnifier;
import org.ml_methods_group.common.preparation.basic.MinValuePicker;
import org.ml_methods_group.common.serialization.MarkedSolutionsClusters;
import org.ml_methods_group.common.serialization.SolutionsDataset;
import org.ml_methods_group.evaluation.approaches.*;
import org.ml_methods_group.marking.markers.ManualClusterMarker;
import org.ml_methods_group.marking.markers.Marker;
import org.ml_methods_group.testing.markers.CacheMarker;
import org.ml_methods_group.testing.markers.CachedExtrapolationMarker;
import org.ml_methods_group.testing.selectors.CacheOptionSelector;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.Function;
import java.util.stream.IntStream;

import static org.ml_methods_group.common.Solution.Verdict.FAIL;
import static org.ml_methods_group.common.Solution.Verdict.OK;

public class ClusterizationEvaluation {

    //        public static int[] sizeBounds = {20, 30, 40, 50};
    public static int[] sizeBounds = {5, 10, 20, 30};

    public static double hacThreshould = 0.1;
//    public static double hacThreshould = 0.2;
//    public static double hacThreshould = 0.3;
//    public static double hacThreshould = 0.4;

        public static String problem = "min_max";
//    public static String problem = "double_sum";
//    public static String problem = "double_equality";
//    public static String problem = "loggers";
//    public static String problem = "deserialization";
//    public static String problem = "factorial";
//    public static String problem = "merge";

    public static void main(String[] args) throws Exception {
        SolutionsDataset train = SolutionsDataset.load(Paths.get(".cache", "datasets", problem, "train.tmp"));
        Database database = new HashDatabase(Paths.get(""));
        final FeaturesExtractor<Solution, Changes> changesExtractor = createChangesExtractor(train, database);
        System.out.println("Extractor created");

        final Approach<BOWVector> approach = BOWApproach.getDefaultApproach(20000, train, changesExtractor);
//        final Approach<List<String>> approach = JaccardApproach.getDefaultApproach(changesExtractor);
//        final Approach<List<String>> approach = JaccardApproach.getExtendedApproach(changesExtractor);
//        final Approach<List<String>> approach = JaccardApproach.getFullApproach(changesExtractor);
//        final Approach<Changes> approach = FuzzyJaccardApproach.getDefaultApproach(changesExtractor);
//        final Approach<List<double[]>> approach = VectorizationApproach.getDefaultApproach(train, changesExtractor);
        System.out.println("Approach initialized");

        final Path path = Paths.get(".cache", "marked_clusters", problem, approach.name);
        path.toFile().mkdirs();
        evaluate(approach, train, database, hacThreshould, sizeBounds, path);
        System.out.println("Evaluation finished");
    }

    public static FeaturesExtractor<Solution, Changes> createChangesExtractor(SolutionsDataset dataset,
                                                                              Database database)
            throws Exception {
        final ASTGenerator generator = new CachedASTGenerator(new NamesASTNormalizer());
        final ChangeGenerator changesGenerator = new BasicChangeGenerator(generator);
        final Unifier<Solution> unifier =
                new BasicUnifier<>(
                        CommonUtils.compose(generator::buildTree, ITree::getHash)::apply,
                        CommonUtils.checkEquals(generator::buildTree, ASTUtils::deepEquals),
                        new MinValuePicker<>(Comparator.comparingInt(Solution::getSolutionId)));
        final DistanceFunction<Solution> metric =
                new HeuristicChangesBasedDistanceFunction(changesGenerator);
        final List<Solution> correct = dataset.getValues(s -> s.getVerdict() == OK);
        final OptionSelector<Solution, Solution> selector = new CacheOptionSelector<>(
                new ClosestPairSelector<>(unifier.unify(correct), metric),
                database,
                Solution::getSolutionId,
                Solution::getSolutionId);
        return new ChangesExtractor(changesGenerator, selector);
    }

    public static <F> void evaluate(Approach<F> approach, SolutionsDataset train, Database database,
                                    double hacThreshold, int[] sizeBounds, Path dir) throws Exception {
        final Clusterer<Solution> clusterer = new CompositeClusterer<>(
                approach.extractor,
                new HAC<Wrapper<F, Solution>>(hacThreshold, 10,
                        CommonUtils.metricFor(approach.metric, Wrapper::getFeatures)));
        final List<Solution> incorrect = train.getValues(x -> x.getVerdict() == FAIL);
        final Clusters<Solution> clusters = clusterer.buildClusters(incorrect);
        final int minBound = Arrays.stream(sizeBounds).min().orElse(0);

        System.out.println(clusters.getClusters().stream().mapToInt(Cluster::size).filter(x -> x >= minBound).count());
        System.out.println(clusters.getClusters().size());


        final Marker<Cluster<Solution>, String> clusterMarker =
                new CachedExtrapolationMarker<>(
                        new CacheMarker<>(Solution::getSolutionId,
                                String.class,
                                Marker.constMarker(null),
                                database),
                        5,
                        new ManualClusterMarker(10));

        final List<Map<Cluster<Solution>, String>> marks = new ArrayList<>();
        IntStream.range(0, sizeBounds.length).forEach(x -> marks.add(new HashMap<>()));
        for (Cluster<Solution> cluster : clusters.getClusters()) {
            if (cluster.size() < minBound) {
                continue;
            }
            final String mark = clusterMarker.mark(cluster);
            if (mark == null) {
                continue;
            }
            for (int i = 0; i < sizeBounds.length; i++) {
                if (cluster.size() >= sizeBounds[i]) {
                    marks.get(i).put(cluster, mark);
                }
            }
        }
        for (int i = 0; i < sizeBounds.length; i++) {
            printReport(incorrect, clusters.getClusters(), sizeBounds[i], marks.get(i));
            final MarkedSolutionsClusters result = new MarkedSolutionsClusters(marks.get(i));
            final Path path = dir.resolve(hacThreshold + "_" + sizeBounds[i] + ".mc");
            result.store(path);
        }
    }

    private static void printReport(List<Solution> solutions, List<Cluster<Solution>> clusters,
                                    int limit, Map<Cluster<Solution>, String> marks) {
        System.out.println("Report for bound: " + limit);
        System.out.println("    Elements count: " + solutions.size());
        System.out.println("    Clusters count: " + clusters.size());
        System.out.println("    Big clusters count: " + clusters.stream()
                .mapToInt(Cluster::size)
                .filter(x -> x >= limit)
                .count());
        System.out.println("    Good clusters count: " + marks.size());
        System.out.println("    Error types count: " + new HashSet<>(marks.values()).size());
        System.out.println("    Coverage: " + marks.keySet().stream().mapToInt(Cluster::size).sum());
        System.out.println();
    }
}
