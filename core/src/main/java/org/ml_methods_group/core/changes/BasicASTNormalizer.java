package org.ml_methods_group.core.changes;

import com.github.gumtreediff.tree.ITree;
import com.github.gumtreediff.tree.TreeContext;
import org.ml_methods_group.core.entities.NodeType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class BasicASTNormalizer implements ASTNormalizer {

    private static final int BLOCK_TYPE_ID = NodeType.BLOCK.ordinal();
    private static final String BLOCK_TYPE_NAME = "Block";

    @Override
    public void normalize(TreeContext context) {
        for (ITree node : context.getRoot().getTrees()) {
            final NodeType type = NodeType.valueOf(node.getType());
            if (type == null) {
                continue;
            }
            switch (type) {
                case IF_STATEMENT:
                    checkBlocks(node, context, 1, 2);
                    break;
                case FOR_STATEMENT:
                    checkBlocks(node, context, 3);
                    break;
                case ENHANCED_FOR_STATEMENT:
                    checkBlocks(node, context, 2);
                    break;
                case WHILE_STATEMENT:
                    checkBlocks(node, context, 1);
                    break;
            }
        }
        context.validate();
    }

    private void checkBlocks(ITree statement, TreeContext context, int... positions) {
        final List<ITree> children = statement.getChildren();
        final boolean problems = Arrays.stream(positions)
                .filter(x -> x < children.size())
                .mapToObj(children::get)
                .mapToInt(ITree::getType)
                .anyMatch(type -> type != BLOCK_TYPE_ID);
        if (problems) {
            final List<ITree> generated = new ArrayList<>(children);
            for (int position : positions) {
                if (position >= children.size()) continue;
                final ITree child = children.get(position);
                if (child.getType() != BLOCK_TYPE_ID) {
                    final ITree wrapper = context.createTree(BLOCK_TYPE_ID, "", BLOCK_TYPE_NAME);
                    wrapper.addChild(child);
                    generated.set(position, wrapper);
                }
            }
            statement.setChildren(generated);
        }
    }
}
