package org.ml_methods_group.evaluation;

import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.cache.HashDatabase;
import org.ml_methods_group.common.*;
import org.ml_methods_group.common.ast.ASTUtils;
import org.ml_methods_group.common.ast.changes.BasicChangeGenerator;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.common.ast.generation.ASTGenerator;
import org.ml_methods_group.common.ast.generation.CachedASTGenerator;
import org.ml_methods_group.common.ast.normalization.NamesASTNormalizer;
import org.ml_methods_group.common.extractors.ChangesExtractor;
import org.ml_methods_group.common.extractors.ManyProblemsBasedChangesExtractor;
import org.ml_methods_group.common.metrics.functions.HeuristicChangesBasedDistanceFunction;
import org.ml_methods_group.common.metrics.selectors.ClosestPairSelector;
import org.ml_methods_group.common.preparation.basic.BasicUnifier;
import org.ml_methods_group.common.preparation.basic.MinValuePicker;
import org.ml_methods_group.common.serialization.ProtobufSerializationUtils;
import org.ml_methods_group.testing.selectors.CacheOptionSelector;

import java.nio.file.Path;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

import static org.ml_methods_group.common.Solution.Verdict.FAIL;
import static org.ml_methods_group.common.Solution.Verdict.OK;
import static org.ml_methods_group.common.serialization.ProtobufSerializationUtils.loadDataset;
import static org.ml_methods_group.common.serialization.ProtobufSerializationUtils.storeSolutionMarksHolder;

public class MarkingEvaluation {

    private final static Path PATH_TO_DATASET = EvaluationInfo.PATH_TO_DATASET.resolve("integral");
    private final static Integer NUM_CLUSTERS = 100;
    private final static Integer NUM_EXAMPLES = 5;

    public static void main(String[] argv) throws Exception {
    }

    public static void markGlobalClusters(List<String> problems) throws Exception {
        try (final var database = new HashDatabase(EvaluationInfo.PATH_TO_CACHE)) {
            final ASTGenerator astGenerator = new CachedASTGenerator(new NamesASTNormalizer());
            final var changeGenerator = new BasicChangeGenerator(astGenerator);
            final var unifier = new BasicUnifier<>(
                    CommonUtils.compose(astGenerator::buildTree, ITree::getHash)::apply,
                    CommonUtils.checkEquals(astGenerator::buildTree, ASTUtils::deepEquals),
                    new MinValuePicker<>(Comparator.comparingInt(Solution::getSolutionId)));
            final var metric = new HeuristicChangesBasedDistanceFunction(changeGenerator);
            final var generatorByProblemId = new HashMap<Integer, FeaturesExtractor<Solution, Changes>>();

            for (String problem : problems) {
                final Dataset train = loadDataset(EvaluationInfo.PATH_TO_DATASET
                        .resolve("full").resolve(problem).resolve("train.tmp"));
                final List<Solution> correct = train.getValues(x -> x.getVerdict() == OK);
                final var selector = new CacheOptionSelector<>(
                        new ClosestPairSelector<>(unifier.unify(correct), metric),
                        database,
                        Solution::getSolutionId,
                        Solution::getSolutionId
                );
                int problemId = train.getValues().get(0).getProblemId();
                generatorByProblemId.put(problemId, new ChangesExtractor(changeGenerator, selector));
            }
            final var generator = new ManyProblemsBasedChangesExtractor(generatorByProblemId);
            final var clusters = ProtobufSerializationUtils
                    .loadSolutionClusters(EvaluationInfo.PATH_TO_CLUSTERS.resolve("global_clusters.tmp"))
                    .getClusters().stream()
                    .sorted(Comparator.<Cluster<Solution>>comparingInt(Cluster::size).reversed())
                    .collect(Collectors.toList());

            final var marks = new HashMap<Cluster<Solution>, String>();
            final var goodClusters = new ArrayList<Cluster<Solution>>();
            try (Scanner scanner = new Scanner(System.in)) {
                for (var cluster : clusters.subList(0, Math.min(clusters.size(), NUM_CLUSTERS))) {
                    System.out.println("Next cluster (size=" + cluster.size() + "):");
                    final List<Solution> solutions = cluster.elementsCopy();
                    Collections.shuffle(solutions);
                    final var problemCounter = solutions.stream()
                            .map(Solution::getProblemId)
                            .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
                    System.out.println("Problems in cluster: ");
                    problemCounter.forEach((k, v) -> System.out.println(k + " - " + v));
                    System.out.println();
                    int start = 0;
                    while (true) {
                        System.out.println("More?");
                        if (scanner.nextLine().equals("+")) {
                            int tmp = start;
                            for (int i = start; i < Math.min(start + NUM_EXAMPLES, solutions.size()); i++) {
                                final var solution = solutions.get(i);
                                System.out.println("    Example #" + i);
                                System.out.println("    Session id: " + solution.getSessionId());
                                System.out.println(solution.getCode());
                                System.out.println();
                                System.out.println("    Submission fix:");
                                generator.process(solution).getChanges().forEach(System.out::println);
                                System.out.println();
                                tmp++;
                            }
                            start = tmp;
                        } else {
                            break;
                        }
                    }
                    System.out.println("---------------------------------------------------------------------------------");
                    System.out.println("Your mark:");
                    while (true) {
                        final String mark = scanner.nextLine();
                        if (mark.equals("-")) {
                            //marks.remove(cluster);
                            break;
                        } else if (mark.equals("+")) {
                            //System.out.println("Final mark: " + marks.get(cluster));
                            goodClusters.add(cluster);
                            break;
                        } else {
                            //marks.put(cluster, mark);
                        }
                    }
                }
            }
            System.out.println(goodClusters.size());
            final var marked = new MarkedClusters<>(marks);
            ProtobufSerializationUtils.storeSolutionClusters(
                    new Clusters(goodClusters),
                    EvaluationInfo.PATH_TO_CLUSTERS.resolve("good_global_clusters.tmp")
            );
        }
    }

    public static void markClusters() throws Exception {
        try (final var database = new HashDatabase(EvaluationInfo.PATH_TO_CACHE)) {
            final ASTGenerator astGenerator = new CachedASTGenerator(new NamesASTNormalizer());
            final var changeGenerator = new BasicChangeGenerator(astGenerator);
            final var unifier = new BasicUnifier<>(
                    CommonUtils.compose(astGenerator::buildTree, ITree::getHash)::apply,
                    CommonUtils.checkEquals(astGenerator::buildTree, ASTUtils::deepEquals),
                    new MinValuePicker<>(Comparator.comparingInt(Solution::getSolutionId)));
            final var metric = new HeuristicChangesBasedDistanceFunction(changeGenerator);

            final Dataset train = loadDataset(PATH_TO_DATASET.resolve("train.tmp"));
            final List<Solution> correct = train.getValues(x -> x.getVerdict() == OK);
            final OptionSelector<Solution, Solution> selector = new CacheOptionSelector<>(
                    new ClosestPairSelector<>(unifier.unify(correct), metric),
                    database,
                    Solution::getSolutionId,
                    Solution::getSolutionId);
            final FeaturesExtractor<Solution, Changes> generator = new ChangesExtractor(changeGenerator, selector);

            final var clusters = ProtobufSerializationUtils
                    .loadSolutionClusters(PATH_TO_DATASET.resolve("unmarked_clusters.tmp"))
                    .getClusters().stream()
                    .sorted(Comparator.<Cluster<Solution>>comparingInt(Cluster::size).reversed())
                    .collect(Collectors.toList());

            final var marks = new HashMap<Cluster<Solution>, String>();
            try (Scanner scanner = new Scanner(System.in)) {
                for (var cluster : clusters.subList(0, Math.min(clusters.size(), NUM_CLUSTERS))) {
                    System.out.println("Next cluster (size=" + cluster.size() + "):");
                    final List<Solution> solutions = cluster.elementsCopy();
                    Collections.shuffle(solutions);
                    int start = 0;
                    while (true) {
                        System.out.println("More?");
                        if (scanner.nextLine().equals("+")) {
                            int tmp = start;
                            for (int i = start; i < Math.min(start + NUM_EXAMPLES, solutions.size()); i++) {
                                final var solution = solutions.get(i);
                                System.out.println("    Example #" + i);
                                System.out.println("    Session id: " + solution.getSessionId());
                                System.out.println(solution.getCode());
                                System.out.println();
                                System.out.println("    Submission fix:");
                                generator.process(solution).getChanges().forEach(System.out::println);
                                System.out.println();
                                tmp++;
                            }
                            start = tmp;
                        } else {
                            break;
                        }
                    }
                    System.out.println("---------------------------------------------------------------------------------");
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
            ProtobufSerializationUtils.storeMarkedClusters(marked, PATH_TO_DATASET.resolve("clusters.tmp"));
        }
    }

    public static void markSolutions() throws Exception {
        try (final var database = new HashDatabase(EvaluationInfo.PATH_TO_CACHE)) {
            final ASTGenerator astGenerator = new CachedASTGenerator(new NamesASTNormalizer());
            final var changeGenerator = new BasicChangeGenerator(astGenerator);
            final var unifier = new BasicUnifier<>(
                    CommonUtils.compose(astGenerator::buildTree, ITree::getHash)::apply,
                    CommonUtils.checkEquals(astGenerator::buildTree, ASTUtils::deepEquals),
                    new MinValuePicker<>(Comparator.comparingInt(Solution::getSolutionId)));
            final var metric = new HeuristicChangesBasedDistanceFunction(changeGenerator);
            final Dataset test = loadDataset(PATH_TO_DATASET.resolve("test.tmp"));
            final List<Solution> incorrect = test.getValues(x -> x.getVerdict() == FAIL);
            final List<Solution> correct = test.getValues(x -> x.getVerdict() == OK);
            final OptionSelector<Solution, Solution> selector = new CacheOptionSelector<>(
                    new ClosestPairSelector<>(unifier.unify(correct), metric),
                    database,
                    Solution::getSolutionId,
                    Solution::getSolutionId);
            final FeaturesExtractor<Solution, Changes> generator = new ChangesExtractor(changeGenerator, selector);

            final Map<Solution, List<String>> marks = new HashMap<>();
            try (Scanner scanner = new Scanner(System.in)) {
                for (var solution : incorrect) {
                    System.out.println("    Session id: " + solution.getSessionId());
                    System.out.println(solution.getCode());
                    System.out.println();
                    System.out.println("    Submission fix:");
                    generator.process(solution).getChanges().forEach(System.out::println);
                    System.out.println();
                    System.out.println("-------------------------------------------------");
                    System.out.println("Your mark:");
                    List<String> curMarks = new ArrayList<>();
                    while (true) {
                        final String mark = scanner.nextLine();
                        if (mark.equals("+")) {
                            break;
                        } else {
                            curMarks.add(mark);
                        }
                    }
                    marks.put(solution, curMarks);
                }
            }
            SolutionMarksHolder holder = new SolutionMarksHolder(marks);
            storeSolutionMarksHolder(holder, PATH_TO_DATASET.resolve("test_marks.tmp"));
        }
    }
}
