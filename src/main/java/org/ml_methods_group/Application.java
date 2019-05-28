package org.ml_methods_group;

import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.classification.classifiers.CompositeClassifier;
import org.ml_methods_group.classification.classifiers.KNearestNeighbors;
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
import org.ml_methods_group.common.extractors.ChangesExtractor;
import org.ml_methods_group.common.metrics.functions.HeuristicChangesBasedDistanceFunction;
import org.ml_methods_group.common.metrics.selectors.ClosestPairSelector;
import org.ml_methods_group.common.metrics.selectors.FixedIdOptionSelector;
import org.ml_methods_group.common.preparation.Unifier;
import org.ml_methods_group.common.preparation.basic.BasicUnifier;
import org.ml_methods_group.common.preparation.basic.MinValuePicker;
import org.ml_methods_group.common.serialization.ProtobufSerializationUtils;
import org.ml_methods_group.evaluation.approaches.BOWApproach;
import org.ml_methods_group.evaluation.approaches.FuzzyJaccardApproach;
import org.ml_methods_group.parsing.JavaCodeValidator;
import org.ml_methods_group.parsing.ParsingUtils;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

import static org.ml_methods_group.common.Solution.Verdict.FAIL;
import static org.ml_methods_group.common.Solution.Verdict.OK;

public class Application {
    public static void main(String[] args) throws IOException {
        switch (args[0]) {
            case "parse":
                parse(Paths.get(args[1]), Paths.get(args[2]));
                break;
            case "cluster":
                cluster(Paths.get(args[1]), Paths.get(args[2]));
                break;
            case "mark":
                mark(Paths.get(args[1]), Paths.get(args[2]), Integer.parseInt(args[3]), Integer.parseInt(args[4]));
                break;
            case "classify":
                classify(Paths.get(args[1]), Paths.get(args[2]), Paths.get(args[3]));
                break;
            default:
                System.out.println("Undefined command!");
        }
    }

    public static void parse(Path data, Path storage) throws IOException {
        try (InputStream input = new FileInputStream(data.toFile())) {
            final Dataset dataset = ParsingUtils.parse(input, new JavaCodeValidator(), x -> true);
            ProtobufSerializationUtils.storeDataset(dataset, storage);
        }
    }

    public static void cluster(Path data, Path storage) throws IOException {
        final Dataset dataset = ProtobufSerializationUtils.loadDataset(data);
        final var selector = new FixedIdOptionSelector<>(
                dataset.getValues(x -> x.getVerdict() == OK),
                Solution::getSessionId,
                Solution::getSessionId);
        final var normalizer = new NamesASTNormalizer();
        final var treeGenerator = new CachedASTGenerator(normalizer);
        final var changeGenerator = new BasicChangeGenerator(treeGenerator);
        final var extractor = new ChangesExtractor(changeGenerator, selector);
        final var approach = BOWApproach.getDefaultApproach(20000, dataset, extractor);
        final var clusterer = new CompositeClusterer<>(approach.extractor, new HAC<>(
                0.3,
                1,
                CommonUtils.metricFor(approach.metric, Wrapper::getFeatures)));
        final var clusters = clusterer.buildClusters(dataset.getValues(x -> x.getVerdict() == FAIL));
        ProtobufSerializationUtils.storeSolutionClusters(clusters, storage);
    }

    public static void mark(Path data, Path dst, int numExamples, int numClusters) throws IOException {
        final var clusters = ProtobufSerializationUtils.loadSolutionClusters(data)
                .getClusters().stream()
                .sorted(Comparator.<Cluster<Solution>>comparingInt(Cluster::size).reversed())
                .collect(Collectors.toList());
        final Map<Cluster<Solution>, String> marks = new HashMap<>();
        try (Scanner scanner = new Scanner(System.in)) {
            for (var cluster : clusters.subList(0, Math.min(clusters.size(), numClusters))) {
                System.out.println("Next cluster (size=" + cluster.size() + "):");
                final var solutions = cluster.elementsCopy();
                Collections.shuffle(solutions);
                for (int i = 0; i < Math.min(numExamples, solutions.size()); i++) {
                    final var solution = solutions.get(i);
                    System.out.println("    Example #" + i);
                    System.out.println("    Session id: " + solution.getSessionId());
                    System.out.println(solution.getCode());
                    System.out.println();
                }
                System.out.println("-------------------------------------------------");
                System.out.println("Your mark:");
                final String mark = scanner.next();
                if (!mark.equals("-")) {
                    marks.put(cluster, mark);
                }
            }
        }
        final MarkedClusters<Solution, String> marked = new MarkedClusters<>(marks);
        ProtobufSerializationUtils.storeMarkedClusters(marked, dst);
    }

    public static void classify(Path data, Path marks, Path element) throws IOException {
        final MarkedClusters<Solution, String> clusters = ProtobufSerializationUtils.loadMarkedClusters(marks);
        final var dataset = ProtobufSerializationUtils.loadDataset(data);
        final ASTGenerator astGenerator = new CachedASTGenerator(new NamesASTNormalizer());
        final ChangeGenerator changeGenerator = new BasicChangeGenerator(astGenerator);
        final Unifier<Solution> unifier = new BasicUnifier<>(
                CommonUtils.compose(astGenerator::buildTree, ITree::getHash)::apply,
                CommonUtils.checkEquals(astGenerator::buildTree, ASTUtils::deepEquals),
                new MinValuePicker<>(Comparator.comparingInt(Solution::getSolutionId)));
        final DistanceFunction<Solution> metric =
                new HeuristicChangesBasedDistanceFunction(changeGenerator);
        final OptionSelector<Solution, Solution> selector = new ClosestPairSelector<>(
                unifier.unify(dataset.getValues(x -> x.getVerdict() == OK)), metric);
        final var extractor = new ChangesExtractor(changeGenerator, selector);
        final var approach = FuzzyJaccardApproach.getDefaultApproach(extractor);
        final var distanceFunction = CommonUtils.metricFor(approach.metric, Wrapper<Changes, Solution>::getFeatures);
        final Classifier<Solution, String> classifier = new CompositeClassifier<>(
                approach.extractor,
                new KNearestNeighbors<>(15, distanceFunction));
        classifier.train(clusters);
        final var code = new String(Files.readAllBytes(element));
        final var solution = new Solution(code, -1, -1, -1, FAIL);
        final var result = classifier.mostProbable(solution);
        System.out.println("Solution:");
        System.out.println(code);
        System.out.println("Result: " + result.getKey() + " (" + result.getValue() + ")");
    }
}
