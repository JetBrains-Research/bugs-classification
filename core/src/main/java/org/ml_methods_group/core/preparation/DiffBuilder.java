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
import java.util.Optional;
import java.util.stream.Collectors;

public class DiffBuilder {
    private final Matchers matchers;
    private final TreeGenerator generator;

    public DiffBuilder() {
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
                .map(DiffBuilder::fromAction)
                .collect(Collectors.toList());
    }

    private static String normalize(String label) {
        return label.replaceAll("[^a-zA-Z0-9.,?:;\\\\<>=!+\\-^@~*/%&|(){}\\[\\]]", "").trim();
    }

    public static AtomicChange fromAction(Action action) {
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

    public static InsertChange fromAction(Insert insert) {
        return new InsertChange(
                insert.getNode().getType(),
                normalize(insert.getNode().getLabel()),
                Optional.of(insert.getNode().getParent()).map(ITree::getType).orElse(0),
                Optional.of(insert.getNode().getParent()).map(ITree::getParent).map(ITree::getType).orElse(0)
        );
    }

    public static DeleteChange fromAction(Delete delete) {
        return new DeleteChange(
                delete.getNode().getType(),
                normalize(delete.getNode().getLabel()),
                Optional.of(delete.getNode().getParent()).map(ITree::getType).orElse(0),
                Optional.of(delete.getNode().getParent()).map(ITree::getParent).map(ITree::getType).orElse(0)
        );
    }

    public static MoveChange fromAction(Move move) {
        return new MoveChange(
                move.getNode().getType(),
                normalize(move.getNode().getLabel()),
                move.getParent().getType(),
                Optional.of(move.getParent()).map(ITree::getParent).map(ITree::getType).orElse(0),
                Optional.of(move.getNode().getParent()).map(ITree::getType).orElse(0),
                Optional.of(move.getNode().getParent()).map(ITree::getParent).map(ITree::getType).orElse(0)
        );
    }

    public static UpdateChange fromAction(Update update) {
        return new UpdateChange(
                update.getNode().getType(),
                normalize(update.getValue()),
                update.getNode().getParent().getType(),
                Optional.of(update.getNode().getParent()).map(ITree::getType).orElse(0),
                normalize(update.getNode().getLabel())
        );
    }
}
