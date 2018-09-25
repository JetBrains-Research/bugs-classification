package org.ml_methods_group.common.changes.generation;

import com.github.gumtreediff.tree.TreeContext;

public interface ASTNormalizer {
    void normalize(TreeContext context, String code);
}
