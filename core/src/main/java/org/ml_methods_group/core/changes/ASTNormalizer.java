package org.ml_methods_group.core.changes;

import com.github.gumtreediff.tree.TreeContext;

public interface ASTNormalizer {
    void normalize(TreeContext  context, String code);
}
