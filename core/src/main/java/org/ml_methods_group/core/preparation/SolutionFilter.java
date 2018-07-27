package org.ml_methods_group.core.preparation;

import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.core.changes.ChangeGenerator;
import org.ml_methods_group.core.entities.NodeType;
import org.ml_methods_group.core.entities.Solution;

import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

public class SolutionFilter {

    private final ChangeGenerator generator;

    public SolutionFilter(ChangeGenerator generator) {
        this.generator = generator;
    }

    public boolean accept(Solution solution) {
        final ITree tree = generator.getTree(solution);
        final List<ITree> blocks = tree.getTrees().stream()
                .filter(node -> node.getType() == NodeType.BLOCK.ordinal())
                .collect(Collectors.toList());
        return !blocks.stream()
                .map(ITree::getChildren)
                .allMatch(Collection::isEmpty);
    }
}
