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
import org.ml_methods_group.common.metrics.functions.HeuristicChangesBasedDistanceFunction;
import org.ml_methods_group.common.metrics.selectors.ClosestPairSelector;
import org.ml_methods_group.common.preparation.basic.BasicUnifier;
import org.ml_methods_group.common.preparation.basic.MinValuePicker;
import org.ml_methods_group.common.serialization.ProtobufSerializationUtils;
import org.ml_methods_group.testing.selectors.CacheOptionSelector;

import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

import static org.ml_methods_group.common.Solution.Verdict.FAIL;
import static org.ml_methods_group.common.Solution.Verdict.OK;
import static org.ml_methods_group.common.serialization.ProtobufSerializationUtils.*;

public class MarkingEvaluation {

    private final static Path PATH_TO_DATASET = EvaluationInfo.PATH_TO_DATASET.resolve("filter");
    private final static Integer NUM_CLUSTERS = 50;
    private final static Integer NUM_EXAMPLES = 10;

    public static void main(String[] argv) throws Exception {
        final var marks = loadSolutionMarksHolder(PATH_TO_DATASET.resolve("test_marks.tmp"));
        final Map<Solution, List<String>> fixedMarks = new HashMap<>();
        try (Scanner scanner = new Scanner(System.in)) {
            for (Map.Entry<Solution, List<String>> entry : marks) {
                List<String> newMarks = new ArrayList<>();
                entry.getValue().forEach(mark -> {
                    String current = mark;
                    if (mark.equals("not_all_catched"))
                        current = "null_input";
                    else if (mark.equals("next_int"))
                        current = "wrong_read";
                    System.out.println(current);
                    String newMark = scanner.nextLine();
                    if (newMark.equals("+"))
                        newMarks.add(current);
                    else
                        newMarks.add(newMark);
                });
                fixedMarks.put(entry.getKey(), newMarks);
            }
            storeSolutionMarksHolder(
                    new SolutionMarksHolder(fixedMarks),
                    PATH_TO_DATASET.resolve("test_marks_fixed.tmp")
            );
        }
    }

    public static void markClusters() throws Exception {
        final var database = new HashDatabase(EvaluationInfo.PATH_TO_CACHE);
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
                for (int i = 0; i < Math.min(NUM_EXAMPLES, solutions.size()); i++) {
                    final var solution = solutions.get(i);
                    System.out.println("    Example #" + i);
                    System.out.println("    Session id: " + solution.getSessionId());
                    System.out.println(solution.getCode());
                    System.out.println();
                    System.out.println("    Submission fix:");
                    generator.process(solution).getChanges().forEach(System.out::println);
                    System.out.println();
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

    public static void markSolutions() throws Exception {
        final var database = new HashDatabase(EvaluationInfo.PATH_TO_CACHE);
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
