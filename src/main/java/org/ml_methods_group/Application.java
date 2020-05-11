package org.ml_methods_group;

import com.github.gumtreediff.matchers.CompositeMatchers;
import com.github.gumtreediff.matchers.MappingStore;
import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.clustering.clusterers.CompositeClusterer;
import org.ml_methods_group.clustering.clusterers.HAC;
import org.ml_methods_group.common.*;
import org.ml_methods_group.common.ast.ASTUtils;
import org.ml_methods_group.common.ast.changes.BasicChangeGenerator;
import org.ml_methods_group.common.ast.changes.ChangeGenerator;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.common.ast.changes.CodeChange;
import org.ml_methods_group.common.ast.generation.ASTGenerator;
import org.ml_methods_group.common.ast.generation.CachedASTGenerator;
import org.ml_methods_group.common.ast.normalization.NamesASTNormalizer;
import org.ml_methods_group.common.extractors.BOWExtractor;
import org.ml_methods_group.common.extractors.BOWExtractor.BOWVector;
import org.ml_methods_group.common.extractors.ChangesExtractor;
import org.ml_methods_group.common.extractors.HashExtractor;
import org.ml_methods_group.common.metrics.functions.HeuristicChangesBasedDistanceFunction;
import org.ml_methods_group.common.metrics.selectors.ClosestPairSelector;
import org.ml_methods_group.common.preparation.Unifier;
import org.ml_methods_group.common.preparation.basic.BasicUnifier;
import org.ml_methods_group.common.preparation.basic.MinValuePicker;
import org.ml_methods_group.common.serialization.ProtobufSerializationUtils;
import org.ml_methods_group.parsing.JavaCodeValidator;
import org.ml_methods_group.parsing.ParsingUtils;
import org.ml_methods_group.testing.extractors.CachedFeaturesExtractor;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

import static org.ml_methods_group.common.Hashers.*;
import static org.ml_methods_group.common.Solution.Verdict.FAIL;
import static org.ml_methods_group.common.Solution.Verdict.OK;

public class Application {
    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            System.out.println("Command expected: parse, cluster or mark");
            return;
        }
        switch (args[0]) {
            case "parse":
                if (args.length != 4) {
                    System.out.println("Wrong number of arguments! Expected:" + System.lineSeparator() +
                            "    Path to .csv file with submissions" + System.lineSeparator() +
                            "    Path to file to store parsed solutions" + System.lineSeparator() +
                            "    ID of problem (integer)" + System.lineSeparator());
                    return;
                }
                parse(Paths.get(args[1]), Paths.get(args[2]), Integer.parseInt(args[3]));
                break;
            case "cluster":
                if (args.length != 3) {
                    System.out.println("Wrong number of arguments! Expected:" + System.lineSeparator() +
                            "    Path to file which store parsed solutions" + System.lineSeparator() +
                            "    Path to file to store clusters" + System.lineSeparator());
                    return;
                }
                cluster(Paths.get(args[1]), Paths.get(args[2]));
                break;
            case "mark":
                if (args.length != 5) {
                    System.out.println("Wrong number of arguments! Expected:" + System.lineSeparator() +
                            "    Path to file which store clusters" + System.lineSeparator() +
                            "    Path to file to store marked clusters" + System.lineSeparator() +
                            "    Number of examples to show" + System.lineSeparator() +
                            "    Number of clusters to mark" + System.lineSeparator());
                    return;
                }
                mark(Paths.get(args[1]), Paths.get(args[2]), Integer.parseInt(args[3]), Integer.parseInt(args[4]));
                break;
            case "prepare":
                if (args.length != 4) {
                    System.out.println("Wrong number of arguments! Expected:" + System.lineSeparator() +
                            "    Path to file which store marks" + System.lineSeparator() +
                            "    Path to file which store marks parsed solutions" + System.lineSeparator() +
                            "    Path to file to store prepared marked data" + System.lineSeparator());
                    return;
                }
                prepare(Paths.get(args[1]), Paths.get(args[2]), Paths.get(args[3]));
                break;
            default:
                System.out.println("Undefined command!");
        }
    }

    public static void parse(Path data, Path storage, int problemId) throws IOException {
        try (InputStream input = new FileInputStream(data.toFile())) {
            final Dataset dataset = ParsingUtils.parse(input, new JavaCodeValidator(), x -> x == problemId);
            ProtobufSerializationUtils.storeDataset(dataset, storage);
        }
    }

    public static void cluster(Path data, Path storage) throws IOException {
        final Dataset dataset = ProtobufSerializationUtils.loadDataset(data);
        final ASTGenerator astGenerator = new CachedASTGenerator(new NamesASTNormalizer());
        final ChangeGenerator changeGenerator = new BasicChangeGenerator(astGenerator);
        final Unifier<Solution> unifier = new BasicUnifier<>(
                CommonUtils.compose(astGenerator::buildTree, ITree::getHash)::apply,
                CommonUtils.checkEquals(astGenerator::buildTree, ASTUtils::deepEquals),
                new MinValuePicker<>(Comparator.comparingInt(Solution::getSolutionId)));
        final OptionSelector<Solution, Solution> selector = new ClosestPairSelector<>(
                unifier.unify(dataset.getValues(CommonUtils.check(Solution::getVerdict, OK::equals))),
                new HeuristicChangesBasedDistanceFunction(changeGenerator));
        final var extractor = new CachedFeaturesExtractor<>(
                new ChangesExtractor(changeGenerator, selector),
                Solution::getSolutionId);
        final var changes = dataset.getValues(CommonUtils.check(Solution::getVerdict, FAIL::equals))
                .stream()
                .map(extractor::process)
                .collect(Collectors.toList());
        final var bowExtractor = getBOWExtractor(20000, changes);
        final Clusterer<Changes> clusterer = new CompositeClusterer<>(bowExtractor, new HAC<>(
                0.3,
                1,
                CommonUtils.metricFor(BOWExtractor::cosineDistance, Wrapper::getFeatures)));
        final var clusters = clusterer.buildClusters(changes);
        ProtobufSerializationUtils.storeChangesClusters(clusters, storage);
    }

    public static void mark(Path data, Path dst, int numExamples, int numClusters) throws IOException {
        final var clusters = ProtobufSerializationUtils.loadChangesClusters(data)
                .getClusters().stream()
                .sorted(Comparator.<Cluster<Changes>>comparingInt(Cluster::size).reversed())
                .collect(Collectors.toList());
        final var marks = new HashMap<Cluster<Changes>, String>();
        try (Scanner scanner = new Scanner(System.in)) {
            for (var cluster : clusters.subList(0, Math.min(clusters.size(), numClusters))) {
                System.out.println("Next cluster (size=" + cluster.size() + "):");
                final var solutions = cluster.elementsCopy();
                Collections.shuffle(solutions);
                for (int i = 0; i < Math.min(numExamples, solutions.size()); i++) {
                    final var solution = solutions.get(i);
                    System.out.println("    Example #" + i);
                    System.out.println("    Session id: " + solution.getOrigin().getSessionId());
                    System.out.println(solution.getOrigin().getCode());
                    System.out.println();
                    System.out.println("    Submission fix:");
                    solution.getChanges().forEach(System.out::println);
                    System.out.println();
                }
                System.out.println("-------------------------------------------------");
                System.out.println("Your mark:");
                while (true) {
                    final String mark = scanner.nextLine();
                    if (mark.equals("-")) {
                        marks.remove(cluster);
                    } else if (mark.equals("+")) {
                        System.out.println("Final mark: " + marks.get(cluster));
                        break;
                    } else {
                        marks.put(cluster, mark);
                    }
                }
            }
        }
        final var marked = new MarkedClusters<>(marks);
        ProtobufSerializationUtils.storeMarkedChangesClusters(marked, dst);
    }

    public static void prepare(Path marks, Path data, Path dst) throws IOException {
        final Dataset dataset = ProtobufSerializationUtils.loadDataset(data);
        final ASTGenerator astGenerator = new CachedASTGenerator(new NamesASTNormalizer());
        final ChangeGenerator changeGenerator = new BasicChangeGenerator(
                astGenerator,
                Collections.singletonList((x, y) -> new CompositeMatchers.ClassicGumtree(x, y, new MappingStore())));
        final Unifier<Solution> unifier = new BasicUnifier<>(
                CommonUtils.compose(astGenerator::buildTree, ITree::getHash)::apply,
                CommonUtils.checkEquals(astGenerator::buildTree, ASTUtils::deepEquals),
                new MinValuePicker<>(Comparator.comparingInt(Solution::getSolutionId)));
        final OptionSelector<Solution, Solution> selector = new ClosestPairSelector<>(
                unifier.unify(dataset.getValues(CommonUtils.check(Solution::getVerdict, OK::equals))),
                new HeuristicChangesBasedDistanceFunction(changeGenerator));
        final var extractor = new CachedFeaturesExtractor<>(
                new ChangesExtractor(changeGenerator, selector),
                Solution::getSolutionId);
        final var changes = ProtobufSerializationUtils.loadMarkedChangesClusters(marks);
        final var prepared = changes.map(change -> extractor.process(change.getOrigin()));
        ProtobufSerializationUtils.storeMarkedChangesClusters(prepared, dst);
    }

    public static void classify(Path data, Path marks, Path element) throws IOException {
//        final MarkedClusters<Solution, String> clusters = ProtobufSerializationUtils.loadMarkedClusters(marks);
//        final var dataset = ProtobufSerializationUtils.loadDataset(data);
//        final ASTGenerator astGenerator = new CachedASTGenerator(new NamesASTNormalizer());
//        final ChangeGenerator changeGenerator = new BasicChangeGenerator(astGenerator);
//        final Unifier<Solution> unifier = new BasicUnifier<>(
//                CommonUtils.compose(astGenerator::buildTree, ITree::getHash)::apply,
//                CommonUtils.checkEquals(astGenerator::buildTree, ASTUtils::deepEquals),
//                new MinValuePicker<>(Comparator.comparingInt(Solution::getSolutionId)));
//        final DistanceFunction<Solution> metric =
//                new HeuristicChangesBasedDistanceFunction(changeGenerator);
//        final OptionSelector<Solution, Solution> selector = new ClosestPairSelector<>(
//                unifier.unify(dataset.getValues(x -> x.getVerdict() == OK)), metric);
//        final var extractor = new ChangesExtractor(changeGenerator, selector);
//        final var approach = FuzzyJaccardApproach.getDefaultApproach(extractor);
//        final var distanceFunction = CommonUtils.metricFor(approach.metric, Wrapper<Changes, Solution>::getFeatures);
//        final Classifier<Solution, String> classifier = new CompositeClassifier<>(
//                approach.extractor,
//                new KNearestNeighbors<>(15, distanceFunction));
//        classifier.train(clusters);
//        final var code = new String(Files.readAllBytes(element));
//        final var solution = new Solution(code, -1, -1, -1, FAIL);
//        final var result = classifier.mostProbable(solution);
//        System.out.println("Solution:");
//        System.out.println(code);
//        System.out.println("Result: " + result.getKey() + " (" + result.getValue() + ")");
        throw new UnsupportedOperationException();
    }

    public static FeaturesExtractor<Changes, BOWVector> getBOWExtractor(int wordsLimit, List<Changes> data) {
        final var weak = HashExtractor.<CodeChange.NodeContext>builder()
                .append("TOC")
                .hashComponent(CodeChange.NodeContext::getNode, TYPE_ONLY_NODE_STATE_HASH)
                .hashComponent(CodeChange.NodeContext::getParent, TYPE_ONLY_NODE_STATE_HASH)
                .hashComponent(CodeChange.NodeContext::getParentOfParent, TYPE_ONLY_NODE_STATE_HASH)
                .build();
        final var javaTypes = HashExtractor.<CodeChange.NodeContext>builder()
                .append("JTC")
                .hashComponent(CodeChange.NodeContext::getNode, JAVA_TYPE_NODE_STATE_HASH)
                .hashComponent(CodeChange.NodeContext::getParent, TYPE_ONLY_NODE_STATE_HASH)
                .hashComponent(CodeChange.NodeContext::getParentOfParent, TYPE_ONLY_NODE_STATE_HASH)
                .build();
        final var full = HashExtractor.<CodeChange.NodeContext>builder()
                .append("FCC")
                .hashComponent(CodeChange.NodeContext::getNode, FULL_NODE_STATE_HASH)
                .hashComponent(CodeChange.NodeContext::getParent, TYPE_ONLY_NODE_STATE_HASH)
                .hashComponent(CodeChange.NodeContext::getParentOfParent, TYPE_ONLY_NODE_STATE_HASH)
                .build();
        final var extended = HashExtractor.<CodeChange.NodeContext>builder()
                .append("ECC")
                .hashComponent(CodeChange.NodeContext::getNode, JAVA_TYPE_NODE_STATE_HASH)
                .hashComponent(CodeChange.NodeContext::getParent, LABEL_NODE_STATE_HASH)
                .hashComponent(CodeChange.NodeContext::getParentOfParent, TYPE_ONLY_NODE_STATE_HASH)
                .build();
        final var fullExtended = HashExtractor.<CodeChange.NodeContext>builder()
                .append("FEC")
                .hashComponent(CodeChange.NodeContext::getNode, FULL_NODE_STATE_HASH)
                .hashComponent(CodeChange.NodeContext::getParent, LABEL_NODE_STATE_HASH)
                .hashComponent(CodeChange.NodeContext::getParentOfParent, TYPE_ONLY_NODE_STATE_HASH)
                .build();
        final var deepExtended = HashExtractor.<CodeChange.NodeContext>builder()
                .append("DEC")
                .hashComponent(CodeChange.NodeContext::getNode, JAVA_TYPE_NODE_STATE_HASH)
                .hashComponent(CodeChange.NodeContext::getParent, LABEL_NODE_STATE_HASH)
                .hashComponent(CodeChange.NodeContext::getParentOfParent, LABEL_NODE_STATE_HASH)
                .build();
        final var codeChanges = data.stream()
                .map(Changes::getChanges)
                .flatMap(List::stream)
                .collect(Collectors.toList());
        final var extractors = Arrays.asList(getCodeChangeHasher(weak),
                getCodeChangeHasher(javaTypes), getCodeChangeHasher(full), getCodeChangeHasher(extended),
                getCodeChangeHasher(fullExtended), getCodeChangeHasher(deepExtended));
        final HashMap<String, Integer> dict = BOWExtractor.mostCommon(
                extractors,
                codeChanges,
                wordsLimit);
        return new BOWExtractor<>(dict, extractors).extend(Changes::getChanges);
    }
}
