package org.ml_methods_group.changes;

import com.github.gumtreediff.actions.ActionGenerator;
import com.github.gumtreediff.actions.model.*;
import com.github.gumtreediff.gen.jdt.JdtTreeGenerator;
import com.github.gumtreediff.matchers.Matcher;
import com.github.gumtreediff.matchers.Matchers;
import com.github.gumtreediff.tree.ITree;

import java.io.IOException;
import java.sql.SQLException;
import java.util.List;
import java.util.stream.Collectors;

public class ChangeUtils {
    static int getNodeType(ITree node) {
        return node == null ? 0 : node.getType() + 1;
    }

    static int getNodeType(ITree node, int step) {
        return step == 0 ? getNodeType(node) : getNodeType(node, step - 1);
    }

    static String normalize(String label) {
        return label.replaceAll("[^a-zA-Z0-9.,?:;\\\\<>=!+\\-^@~*/%&|(){}\\[\\]]", "").trim();
    }

    public static AtomicChange fromAction(Action action) {
        if (action.getClass() == Insert.class) {
            return InsertChange.fromAction((Insert) action);
        } else if (action.getClass() == Delete.class) {
            return DeleteChange.fromAction((Delete) action);
        } else if (action.getClass() == Move.class) {
            return MoveChange.fromAction((Move) action);
        } else if (action.getClass() == Update.class) {
            return UpdateChange.fromAction((Update) action);
        }
        throw new RuntimeException("Unexpected action type!");
    }

    public static AtomicChange fromData(Object[] data, int offset) throws SQLException {
        final int changeType = (Integer) data[1 + AtomicChange.CHANGE_TYPE_OFFSET];
        if (changeType == ChangeType.INSERT.ordinal()) {
            return InsertChange.fromData(data, offset);
        } else if (changeType == ChangeType.DELETE.ordinal()) {
            return DeleteChange.fromData(data, offset);
        } else if (changeType == ChangeType.MOVE.ordinal()) {
            return MoveChange.fromData(data, offset);
        } else if (changeType == ChangeType.UPDATE.ordinal()) {
            return UpdateChange.fromData(data, offset);
        }
        throw new RuntimeException("Unexpected action type!");
    }

    public static List<AtomicChange> calculateChanges(String before, String after) throws IOException {
        if (!before.contains("class ") && !after.contains("class ")) {
            before = "public class MY_MAGIC_CLASS_NAME {\n" + before + "\n}";
            after = "public class MY_MAGIC_CLASS_NAME {\n" + after + "\n}";
        }
        final ITree treeBefore = new JdtTreeGenerator().generateFromString(before).getRoot();
        final ITree treeAfter = new JdtTreeGenerator().generateFromString(after).getRoot();
        final Matcher matcher = Matchers.getInstance().getMatcher(treeBefore, treeAfter);
        matcher.match();
        final ActionGenerator g = new ActionGenerator(treeBefore, treeAfter, matcher.getMappings());
        return g.generate().stream()
                .map(ChangeUtils::fromAction)
                .collect(Collectors.toList());
    }
}
