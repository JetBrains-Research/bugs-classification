package org.ml_methods_group.common.metrics.functions;

import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.common.DistanceFunction;
import org.ml_methods_group.common.ast.changes.ChangeGenerator;

public class EditDistance implements DistanceFunction<ITree> {

    private final ChangeGenerator generator;

    public EditDistance(ChangeGenerator generator) {
        this.generator = generator;
    }

    @Override
    public double distance(ITree first, ITree second) {
        return generator.diffSize(first, second);
    }
}
