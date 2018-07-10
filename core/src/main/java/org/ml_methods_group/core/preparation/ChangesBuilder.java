package org.ml_methods_group.core.preparation;

import com.github.gumtreediff.actions.ActionGenerator;
import com.github.gumtreediff.actions.model.*;
import com.github.gumtreediff.gen.TreeGenerator;
import com.github.gumtreediff.gen.jdt.JdtTreeGenerator;
import com.github.gumtreediff.matchers.Matcher;
import com.github.gumtreediff.matchers.Matchers;
import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.core.changes.*;

import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

public class ChangesBuilder {
    private final Matchers matchers;
    private final TreeGenerator generator;

    public ChangesBuilder() {
        matchers = Matchers.getInstance();
        generator = new JdtTreeGenerator();
    }

    public List<AtomicChange> findChanges(String before, String after) throws IOException {
        if (!before.contains("class ") && !after.contains("class ")) {
            before = "public class MY_MAGIC_CLASS_NAME {\n" + before + "\n}";
            after = "public class MY_MAGIC_CLASS_NAME {\n" + after + "\n}";
        }
        final ITree treeBefore = generator.generateFromString(before).getRoot();
        final ITree treeAfter = generator.generateFromString(after).getRoot();
        final Matcher matcher = matchers.getMatcher(treeBefore, treeAfter);
        matcher.match();
        final ActionGenerator actions = new ActionGenerator(treeBefore, treeAfter, matcher.getMappings());
        return actions.generate().stream()
                .map(ChangesBuilder::fromAction)
                .collect(Collectors.toList());
    }

    private static String normalize(String label) {
        return label.replaceAll("[^a-zA-Z0-9.,?:;\\\\<>=!+\\-^@~*/%&|(){}\\[\\]]", "").trim();
    }

    private static AtomicChange fromAction(Action action) {
        if (action.getClass() == Insert.class) {
            return fromAction((Insert) action);
        } else if (action.getClass() == Delete.class) {
            return fromAction((Delete) action);
        } else if (action.getClass() == Move.class) {
            return fromAction((Move) action);
        } else {
            return fromAction((Update) action);
        }
    }

    private static InsertChange fromAction(Insert insert) {
        final ITree node = insert.getNode();
        return new InsertChange(
                getNodeType(node, 0),
                getNodeType(node, 1),
                getNodeType(node, 2),
                normalize(node.getLabel())
        );
    }

    private static DeleteChange fromAction(Delete delete) {
        final ITree node = delete.getNode();
        return new DeleteChange(
                getNodeType(node, 0),
                getNodeType(node, 1),
                getNodeType(node, 2),
                normalize(node.getLabel())
        );
    }

    private static MoveChange fromAction(Move move) {
        final ITree node = move.getNode();
        final ITree parent = move.getParent();
        return new MoveChange(
                getNodeType(node, 0),
                getNodeType(parent, 0),
                getNodeType(parent, 1),
                getNodeType(node, 1),
                getNodeType(node, 2),
                normalize(node.getLabel())
        );
    }

    private static UpdateChange fromAction(Update update) {
        final ITree node = update.getNode();
        return new UpdateChange(
                getNodeType(node, 0),
                getNodeType(node, 1),
                getNodeType(node, 2),
                normalize(update.getValue()),
                normalize(node.getLabel())
        );
    }

    private static NodeType getNodeType(ITree node, int steps) {
        if (node == null) {
            return NodeType.NONE;
        }
        if (steps == 0) {
            return node.getType() < 0 ? NodeType.NONE : NodeType.valueOf(node.getType());
        }
        return getNodeType(node.getParent(), steps - 1);
    }
}
