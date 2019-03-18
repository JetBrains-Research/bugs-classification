package org.ml_methods_group.evaluation;

import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.cache.HashDatabase;
import org.ml_methods_group.common.*;
import org.ml_methods_group.common.ast.ASTUtils;
import org.ml_methods_group.common.ast.changes.BasicChangeGenerator;
import org.ml_methods_group.common.ast.changes.ChangeGenerator;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.common.ast.changes.CodeChange;
import org.ml_methods_group.common.ast.generation.ASTGenerator;
import org.ml_methods_group.common.ast.generation.CachedASTGenerator;
import org.ml_methods_group.common.ast.normalization.NamesASTNormalizer;
import org.ml_methods_group.common.extractors.ChangesExtractor;
import org.ml_methods_group.common.metrics.functions.HeuristicChangesBasedDistanceFunction;
import org.ml_methods_group.common.metrics.selectors.ClosestPairSelector;
import org.ml_methods_group.common.preparation.Unifier;
import org.ml_methods_group.common.preparation.basic.BasicUnifier;
import org.ml_methods_group.common.preparation.basic.MinValuePicker;
import org.ml_methods_group.common.serialization.SolutionsDataset;
import org.ml_methods_group.testing.selectors.CacheOptionSelector;

import java.nio.file.Paths;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Optional;
import java.util.Scanner;

public class TestHelper {

    private static String problem = "double_equality";

    public static void main(String[] args) throws Exception {
        SolutionsDataset test = SolutionsDataset.load(Paths.get(".cache", "datasets", problem, "reserve.tmp"));

        final ASTGenerator generator = new CachedASTGenerator(new NamesASTNormalizer());
        final ChangeGenerator changesGenerator = new BasicChangeGenerator(generator);
        final Unifier<Solution> unifier =
                new BasicUnifier<>(
                        CommonUtils.compose(generator::buildTree, ITree::getHash)::apply,
                        CommonUtils.checkEquals(generator::buildTree, ASTUtils::deepEquals),
                        new MinValuePicker<>(Comparator.comparingInt(Solution::getSolutionId)));
        final DistanceFunction<Solution> metric =
                new HeuristicChangesBasedDistanceFunction(changesGenerator);
        final Database database = new HashDatabase(EvaluationInfo.pathToCache);
        final OptionSelector<Solution, Solution> selector = new CacheOptionSelector<>(
                new ClosestPairSelector<>(unifier.unify(test.getValues(
                        solution -> solution.getVerdict() == Solution.Verdict.OK)), metric),
                database,
                Solution::getSolutionId,
                Solution::getSolutionId);
        final ChangesExtractor extractor = new ChangesExtractor(changesGenerator, selector);

        HashMap<Integer, Solution> good = new HashMap<>();
        test.getValues(x -> x.getVerdict() == Solution.Verdict.OK)
                .forEach(x -> good.put(x.getSessionId(), x));
        HashMap<Integer, Solution> bad = new HashMap<>();
        test.getValues(x -> x.getVerdict() == Solution.Verdict.FAIL)
                .forEach(x -> bad.put(x.getSessionId(), x));
        Scanner scanner = new Scanner(System.in);
        while (true) {
            int session = scanner.nextInt();
            System.out.println(session);
            System.out.println(bad.get(session).getCode());
            System.out.println("-------------------------");
            System.out.println(Optional.ofNullable(good.get(session)).map(Solution::getCode).orElse("NOT_FOUND"));
            System.out.println("-------------------------");
            System.out.println(selector.selectOption(bad.get(session)).get().getCode());
            System.out.println("==========================");
            final Changes changes = extractor.process(bad.get(session));
            for (CodeChange change : changes.getChanges()) {
                System.out.print(change.getChangeType() + " ");
                System.out.print(change.getOriginalContext().getNode().getType() + " ");
                System.out.print(change.getOriginalContext().getNode().getLabel() + " ");
                System.out.print(change.getOriginalContext().getNode().getJavaType() + " ");
                System.out.print(change.getDestinationContext().getNode().getLabel() + " ");
                System.out.print(change.getDestinationContext().getNode().getJavaType() + " ");
                System.out.print(change.getOriginalContext().getParent().getType() + " ");
                System.out.print(change.getOriginalContext().getParent().getLabel() + " ");
                System.out.print(change.getDestinationContext().getParent().getType() + " ");
                System.out.println(change.getDestinationContext().getParent().getLabel());
            }
        }
    }
}
