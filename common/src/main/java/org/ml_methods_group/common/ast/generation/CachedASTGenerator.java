package org.ml_methods_group.common.ast.generation;

import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.ast.normalization.ASTNormalizer;

import java.lang.ref.SoftReference;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class CachedASTGenerator extends BasicASTGenerator {

    private final Map<Solution, SoftReference<ITree>> cache = new ConcurrentHashMap<>();

    public CachedASTGenerator(ASTNormalizer normalizer) {
        super(normalizer);
    }

    public CachedASTGenerator() {
    }

    @Override
    public ITree buildTree(Solution solution) {
        if (solution.getSolutionId() == -1) {
            return super.buildTree(solution);
        }
        final SoftReference<ITree> reference = cache.get(solution);
        final ITree cached = reference == null ? null : reference.get();
        if (cached != null) {
            return cached.deepCopy();
        }
        final ITree tree = super.buildTree(solution);
        cache.put(solution, new SoftReference<>(tree));
        return tree.deepCopy();
    }
}
