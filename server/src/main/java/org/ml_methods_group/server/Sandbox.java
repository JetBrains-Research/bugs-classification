package org.ml_methods_group.server;

import com.github.gumtreediff.matchers.CompositeMatchers;
import com.github.gumtreediff.matchers.MappingStore;
import com.github.gumtreediff.matchers.Matcher;
import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.common.CommonUtils;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.ast.ASTUtils;
import org.ml_methods_group.common.ast.changes.BasicChangeGenerator;
import org.ml_methods_group.common.ast.changes.ChangeGenerator;
import org.ml_methods_group.common.ast.generation.CachedASTGenerator;
import org.ml_methods_group.common.ast.normalization.NamesASTNormalizer;
import org.ml_methods_group.common.preparation.Unifier;
import org.ml_methods_group.common.preparation.basic.BasicUnifier;
import org.ml_methods_group.common.preparation.basic.MinValuePicker;

import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.BiFunction;

import static org.ml_methods_group.common.Solution.Verdict.FAIL;
import static org.ml_methods_group.common.Solution.Verdict.OK;
import static org.ml_methods_group.common.serialization.ProtobufSerializationUtils.loadDataset;

public class Sandbox {
    public static void main(String[] args) throws IOException {
//        53619
        final var correct = loadDataset(
                Paths.get("C:\\internship\\bugs-classification\\data\\58088\\solutions.tmp"))
                .filter(CommonUtils.check(Solution::getVerdict, OK::equals));
        final var incorrect = loadDataset(
                Paths.get("C:\\internship\\bugs-classification\\data\\58088\\solutions.tmp"))
                .filter(CommonUtils.check(Solution::getVerdict, FAIL::equals));
        final var treeGenerator = new CachedASTGenerator(new NamesASTNormalizer());
        final Unifier<Solution> unifier = new BasicUnifier<>(
                CommonUtils.compose(treeGenerator::buildTree, ITree::getHash)::apply,
                CommonUtils.checkEquals(treeGenerator::buildTree, ASTUtils::deepEquals),
                new MinValuePicker<>(Comparator.comparingInt(Solution::getSolutionId)));
        final List<Solution> options = unifier.unify(correct.getValues());
        final var generator1 = new BasicChangeGenerator(treeGenerator,
                Arrays.asList(
                        (Serializable & BiFunction<ITree, ITree, Matcher>) (x, y) ->
                                new CompositeMatchers.CompleteGumtreeMatcher(x, y, new MappingStore())
                ));
        final var generator2 = new BasicChangeGenerator(treeGenerator,
                Arrays.asList(
                        (Serializable & BiFunction<ITree, ITree, Matcher>) (x, y) ->
                                new CompositeMatchers.ClassicGumtree(x, y, new MappingStore())
                ));
        final var generator3 = new BasicChangeGenerator(treeGenerator,
                Arrays.asList(
                        (Serializable & BiFunction<ITree, ITree, Matcher>) (x, y) ->
                                new CompositeMatchers.ChangeDistiller(x, y, new MappingStore())
                ));
        final var generator4 = new BasicChangeGenerator(treeGenerator,
                Arrays.asList(
                        (Serializable & BiFunction<ITree, ITree, Matcher>) (x, y) ->
                                new CompositeMatchers.XyMatcher(x, y, new MappingStore())
                ));
        ChangeGenerator[] generators = {generator1, generator2, generator3, generator4};
        int[] w = new int[4];
        int[] s = new int[4];
        int[] l = new int[4];
        int[] f = new int[4];
        int[] bw = new int[4];
        int total = 0;
        int cnt = 0;
        final List<Solution> solutions = new ArrayList<>(incorrect.getValues());
        Collections.shuffle(solutions);
        for (Solution src : solutions) {
            cnt++;
            int[] best = {Integer.MAX_VALUE, Integer.MAX_VALUE, Integer.MAX_VALUE, Integer.MAX_VALUE};
            for (Solution dst : options) {
                total++;
                final int[] r = new int[4];
                for (int i = 0; i < r.length; i++) {
                    r[i] = test(generators[i], src, dst);
                    if (r[i] == Integer.MAX_VALUE) {
                        f[i]++;
                    } else {
                        s[i] += r[i];
                        best[i] = Math.min(best[i], r[i]);
                    }
                }
                for (int i = 0; i < r.length; i++) {
                    int lose = 0;
                    for (int j = 0; j < r.length; j++) {
                        if (r[i] - r[j] > lose) {
                            lose = r[i] - r[j];
                        }
                    }
                    if (lose == 0) {
                        w[i]++;
                    }
                    l[i] += lose;
                }
//                System.out.println(src.getSessionId() + " " + dst.getSessionId() + " " + resToString(r[0]) + "\t" + resToString(r[1]) + "\t"
//                        + resToString(r[2]) + "\t" + resToString(r[3]));
            }
            for (int i = 0; i < 4; i++) {
                boolean flag = true;
                for (int j = 0; j < 4; j++) {
                    if (best[i] > best[j]) {
                        flag = false;
                        break;
                    }
                }
                if (flag) {
                    bw[i]++;
                }
            }
            System.out.println("Done");
            if (cnt % 100 == 0) {
                double[] a = new double[4];
                double[] p = new double[4];
                double[] tl = new double[4];
                double[] tbw = new double[4];
                for (int i = 0; i < 4; i++) {
                    a[i] = (double) s[i] / (total - f[i]);
                    p[i] = (double) w[i] / total;
                    tl[i] = (double) Math.round(100 * (double) l[i] / total) / 100;
                    tbw[i] = (double) bw[i] / cnt;
                }
                System.out.println("Win count: " + w[0] + " " + w[1] + " " + w[2] + " " + w[3]);
                System.out.println("Win part:  " + (int) (p[0] * 100 + 0.5) + " " + (int) (p[1] * 100 + 0.5) + " "
                        + (int) (p[2] * 100 + 0.5) + " " + (int) (p[3] * 100 + 0.5));
                System.out.println("Best win part:  " + (int) (tbw[0] * 100 + 0.5) + " " + (int) (tbw[1] * 100 + 0.5) + " "
                        + (int) (tbw[2] * 100 + 0.5) + " " + (int) (tbw[3] * 100 + 0.5));
                System.out.println(f[0] + " " + f[1] + " " + f[2] + " " + f[3]);
                System.out.println((int) (a[0] + 0.5) + " " + (int) (a[1] + 0.5) + " " + (int) (a[2] + 0.5) + " "
                        + (int) (a[3] + 0.5));
                System.out.println(tl[0] + " " +tl[1] + " " + tl[2] + " " + tl[3]);
            }
        }
    }

    static int test(ChangeGenerator generator, Solution src, Solution dst) {
        try {
            return generator.getChanges(src, dst).getChanges().size();
        } catch (Exception e) {
            return Integer.MAX_VALUE;
        }
    }

    static String resToString(int value) {
        return value == Integer.MAX_VALUE ? "NO" : Integer.toString(value);
    }
}
