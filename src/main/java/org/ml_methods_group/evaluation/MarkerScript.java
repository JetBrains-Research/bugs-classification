package org.ml_methods_group.evaluation;

import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.cache.HashDatabase;
import org.ml_methods_group.common.*;
import org.ml_methods_group.common.ast.ASTUtils;
import org.ml_methods_group.common.ast.changes.BasicChangeGenerator;
import org.ml_methods_group.common.ast.changes.ChangeGenerator;
import org.ml_methods_group.common.ast.generation.ASTGenerator;
import org.ml_methods_group.common.ast.generation.CachedASTGenerator;
import org.ml_methods_group.common.ast.normalization.NamesASTNormalizer;
import org.ml_methods_group.common.metrics.functions.HeuristicChangesBasedDistanceFunction;
import org.ml_methods_group.common.metrics.selectors.ClosestPairSelector;
import org.ml_methods_group.common.preparation.Unifier;
import org.ml_methods_group.common.preparation.basic.BasicUnifier;
import org.ml_methods_group.common.preparation.basic.MinValuePicker;
import org.ml_methods_group.common.serialization.ProtobufSerializationUtils;
import org.ml_methods_group.testing.selectors.CacheOptionSelector;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

import static org.ml_methods_group.common.Solution.Verdict.FAIL;
import static org.ml_methods_group.common.Solution.Verdict.OK;

public class MarkerScript {
    public static void main(String[] args) throws Exception {
        try (var database = new HashDatabase(EvaluationInfo.PATH_TO_CACHE);
             var input = new Scanner(System.in)) {
            System.out.println("Start marking script");
            System.out.print("Enter problem name: ");
            final String name = input.nextLine().trim();
            System.out.print("Enter dataset name: ");
            final String datasetType = input.nextLine().trim();
            final Path folder = Paths.get(".cache", "datasets", name);
            final Path marksPath = folder.resolve(datasetType + "_marks.tmp");
            final Dataset dataset = ProtobufSerializationUtils.loadDataset(folder.resolve(datasetType + ".tmp"));
            final SolutionMarksHolder holder;

            SolutionMarksHolder tmp;
            try {
                tmp = ProtobufSerializationUtils.loadSolutionMarksHolder(marksPath);
            } catch (IOException e) {
                tmp = new SolutionMarksHolder();
            }
            holder = tmp;

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
                    database, Solution::getSolutionId,
                    Solution::getSolutionId);

            final var permutation = new ArrayList<>(dataset.getValues(x -> x.getVerdict() == FAIL));
            Collections.shuffle(permutation);
            System.out.print("Enter number of solutions: ");
            int counter = Integer.parseInt(input.nextLine().trim());
            loop:
            for (Solution solution : permutation) {
                if (holder.getMarks(solution).isPresent()) {
                    continue;
                }
                System.out.println("Next submission:");
                var closest = selector.selectOption(solution).orElseGet(Solution::new);
                System.out.println("Closest correct submission:");
                System.out.println(closest.getSessionId());
                System.out.println(closest.getCode());
                System.out.println("Submission:");
                System.out.println(solution.getSessionId());
                System.out.println(solution.getCode());
                System.out.println("Changes:");
                changesGenerator.getChanges(solution, closest).getChanges()
                        .forEach(System.out::println);
                while (true) {
                    System.out.print("Your marks:");
                    final String[] marks = input.nextLine().trim().split("\\s+");
                    System.out.println("Accept: " + String.join(" | ", marks) + "?");
                    final String verdict = input.nextLine().trim();
                    if (verdict.equals("+")) {
                        Arrays.stream(marks).forEachOrdered(x -> holder.addMark(solution, x));
                        break;
                    } else if (verdict.equals("break")) {
                        break loop;
                    }
                }
                if (--counter == 0) {
                    break;
                }
            }
            ProtobufSerializationUtils.storeSolutionMarksHolder(holder, marksPath);
        }
    }
}
