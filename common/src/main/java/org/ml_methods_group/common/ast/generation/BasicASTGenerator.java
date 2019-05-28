package org.ml_methods_group.common.ast.generation;

import com.github.gumtreediff.gen.TreeGenerator;
import com.github.gumtreediff.gen.jdt.JdtTreeGenerator;
import com.github.gumtreediff.tree.ITree;
import com.github.gumtreediff.tree.TreeContext;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.ast.normalization.ASTNormalizer;

import java.io.IOException;

public class BasicASTGenerator implements ASTGenerator {

    private final ASTNormalizer normalizer;
    private final TreeGenerator generator;

    public BasicASTGenerator(ASTNormalizer normalizer) {
        this.normalizer = normalizer;
        generator = new JdtTreeGenerator();
    }

    public BasicASTGenerator() {
        this(null);
    }

    @Override
    public ITree buildTree(Solution solution) {
        try {
            final String code = solution.getCode();
            final TreeContext context;
            context = generator.generateFromString(code);
            if (normalizer != null) {
                normalizer.normalize(context, code);
            }
            return context.getRoot();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
