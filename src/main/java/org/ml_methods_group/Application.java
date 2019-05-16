package org.ml_methods_group;

import org.ml_methods_group.common.*;
import org.ml_methods_group.common.ast.changes.BasicChangeGenerator;
import org.ml_methods_group.common.ast.generation.CachedASTGenerator;
import org.ml_methods_group.common.ast.normalization.BasicASTNormalizer;
import org.ml_methods_group.common.ast.normalization.NamesASTNormalizer;
import org.ml_methods_group.common.extractors.ChangesExtractor;
import org.ml_methods_group.common.metrics.selectors.FixedIdOptionSelector;
import org.ml_methods_group.common.serialization.ProtobufSerializationUtils;
import org.ml_methods_group.evaluation.approaches.BOWApproach;
import org.ml_methods_group.evaluation.approaches.FuzzyJaccardApproach;
import org.ml_methods_group.evaluation.approaches.JaccardApproach;
import org.ml_methods_group.evaluation.approaches.clustering.ClusteringApproachTemplate;
import org.ml_methods_group.parsing.JavaCodeValidator;
import org.ml_methods_group.parsing.ParsingUtils;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.Predicate;
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
                cluster(Paths.get(args[1]), Paths.get(args[2]), APPROACHES.get(args[3]),
                        Double.parseDouble(args[4]), Boolean.parseBoolean(args[5]));
                break;
            case "show":
                show(Paths.get(args[1]), Paths.get(args[2]), Integer.parseInt(args[3]), Integer.parseInt(args[4]));
                break;
            default:
                System.out.println("Undefined command!");
        }
    }

    public static Map<String, ClusteringApproachTemplate> APPROACHES = new HashMap<>() {{
        put("def_jac", new ClusteringApproachTemplate((dataset, extractor) ->
                JaccardApproach.getDefaultApproach(extractor)));
        put("ext_jac", new ClusteringApproachTemplate((dataset, extractor) ->
                JaccardApproach.getExtendedApproach(extractor)));
        put("ful_jac", new ClusteringApproachTemplate((dataset, extractor) ->
                JaccardApproach.getFullApproach(extractor)));
        put("fuz_jac", new ClusteringApproachTemplate((dataset, extractor) ->
                FuzzyJaccardApproach.getDefaultApproach(extractor)));
        put("bow", new ClusteringApproachTemplate((dataset, extractor) ->
                BOWApproach.getDefaultApproach(20000, dataset, extractor)));
    }};

    public static void parse(Path data, Path storage) throws IOException {
        try (InputStream input = new FileInputStream(data.toFile())) {
            final Dataset dataset = ParsingUtils.parse(input, new JavaCodeValidator(), x -> true);
            ProtobufSerializationUtils.storeDataset(dataset, storage);
        }
    }

    public static void cluster(Path data, Path storage,
                               ClusteringApproachTemplate template,
                               double threshold, boolean changeNames) throws IOException {
        final Dataset dataset = ProtobufSerializationUtils.loadDataset(data);
        final var selector = new FixedIdOptionSelector<>(
                dataset.getValues(x -> x.getVerdict() == OK),
                Solution::getSessionId,
                Solution::getSessionId);
        final var normalizer = changeNames ? new NamesASTNormalizer() : new BasicASTNormalizer();
        final var treeGenerator = new CachedASTGenerator(normalizer);
        final var changeGenerator = new BasicChangeGenerator(treeGenerator);
        final var extractor = new ChangesExtractor(changeGenerator, selector);
        final var approach = template.createApproach(
                dataset.filter(x -> selector.selectOption(x).isPresent()),
                extractor);
        final var clusterer = approach.getClusterer(threshold);
        final Predicate<Solution> checker = x -> x.getVerdict() == FAIL && selector.selectOption(x).isPresent();
        final var clusters = clusterer.buildClusters(dataset.getValues(checker));
        ProtobufSerializationUtils.storeSolutionClusters(clusters, storage);

        System.out.println("Total number of clusters: " + clusters.getClusters().size());
        System.out.println("Total number of clusters (size >= 5): " + clusters.getClusters().stream()
                .filter(x -> x.size() >= 5).count());
        System.out.println("Total number of clusters (size >= 10): " + clusters.getClusters().stream()
                .filter(x -> x.size() >= 10).count());
        System.out.println("Total number of clusters (size >= 15): " + clusters.getClusters().stream()
                .filter(x -> x.size() >= 15).count());
    }

    public static void show(Path data, Path clustersStorage, int numExamples, int numClusters) throws IOException {
        final Dataset dataset = ProtobufSerializationUtils.loadDataset(data);
        final var selector = new FixedIdOptionSelector<>(
                dataset.getValues(x -> x.getVerdict() == OK),
                Solution::getSessionId,
                Solution::getSessionId);
        final var clusters = ProtobufSerializationUtils.loadSolutionClusters(clustersStorage)
                .getClusters().stream()
                .sorted(Comparator.<Cluster<Solution>>comparingInt(Cluster::size).reversed())
                .collect(Collectors.toList());
        try (Scanner scanner = new Scanner(System.in)) {
            for (var cluster : clusters.subList(0, Math.min(clusters.size(), numClusters))) {
                System.out.println("Next cluster (size=" + cluster.size() + "):");
                final var solutions = cluster.elementsCopy();
                Collections.shuffle(solutions);
                for (int i = 0; i < Math.min(numExamples, solutions.size()); i++) {
                    final var solution = solutions.get(i);
                    System.out.println("    Example #" + i);
                    System.out.println("    Session id: " + solution.getSessionId());
                    System.out.println();
                    System.out.println("        before:");
                    System.out.println();
                    System.out.println(solution.getCode());
                    System.out.println("        after:");
                    System.out.println(selector.selectOption(solution).map(Solution::getCode).orElse("*not found*"));
                }
                System.out.println();
                scanner.nextLine();
            }
        }
    }
}
