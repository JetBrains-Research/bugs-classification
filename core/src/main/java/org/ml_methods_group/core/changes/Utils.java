package org.ml_methods_group.core.changes;

import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.core.entities.NodeType;

import static org.ml_methods_group.core.entities.NodeType.*;

public class Utils {
    static boolean isMethodName(ITree node) {
        final ITree parent = node.getParent();
        if (NodeType.valueOf(node.getType()) != SIMPLE_NAME || parent == null) {
            return false;
        }
        final NodeType parentType = NodeType.valueOf(parent.getType());
        final int position = parent.getChildPosition(node);
        if (parentType == METHOD_DECLARATION) {
            return true;
        } else if (parentType == METHOD_REF || parentType == TYPE_METHOD_REFERENCE
                || parentType == EXPRESSION_METHOD_REFERENCE) {
            return position == 1;
        } else if (parentType == METHOD_INVOCATION) {
            if (parent.getChildren().size() == 1) {
                return true;
            }
            for (int i = 1; i < position; i++) {
                if (parent.getChild(i).getType() == SIMPLE_NAME.ordinal()) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }
}
