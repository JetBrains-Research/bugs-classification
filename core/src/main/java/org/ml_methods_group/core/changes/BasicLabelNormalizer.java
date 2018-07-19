package org.ml_methods_group.core.changes;

import com.github.gumtreediff.actions.model.Delete;
import com.github.gumtreediff.actions.model.Insert;
import com.github.gumtreediff.actions.model.Move;
import com.github.gumtreediff.actions.model.Update;
import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.core.entities.NodeType;

public class BasicLabelNormalizer implements LabelNormalizer {
    @Override
    public String normalize(String label, Insert insert) {
        return normalize(label, insert.getNode());
    }

    @Override
    public String normalize(String label, Move move) {
        return normalize(label, move.getNode());
    }

    @Override
    public String normalize(String label, Update update) {
        return normalize(label, update.getNode());
    }

    @Override
    public String normalize(String label, Delete delete) {
        return normalize(label, delete.getNode());
    }

    private String normalize(String label, ITree node) {
        label = validate(label);
        if (label.isEmpty()) {
            return label;
        }
        if (node.getType() == NodeType.QUALIFIED_NAME.ordinal()) {
            return node.getParent().getType() == NodeType.SIMPLE_TYPE.ordinal() ? label : lastName(label);
        } else if (node.getType() == NodeType.SIMPLE_TYPE.ordinal()) {
            return Utils.isMethodName(node) ? label : "";
        }
        return label;
    }

    private String validate(String label) {
        return label.replaceAll("[^a-zA-Z0-9.,?:;\\\\<>=!+\\-^@~*/%&|(){}\\[\\]]", "").trim();
    }

    private String lastName(String label) {
        return label.substring(label.lastIndexOf('.') + 1);
    }
}
