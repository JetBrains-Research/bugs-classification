package org.ml_methods_group.evaluation.vectorization;

import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.common.Dataset;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.ast.changes.CodeChange.NodeContext;
import org.ml_methods_group.common.ast.changes.CodeChange.NodeState;
import org.ml_methods_group.common.ast.changes.MetadataKeys;
import org.ml_methods_group.common.ast.generation.ASTGenerator;
import org.ml_methods_group.common.ast.generation.BasicASTGenerator;
import org.ml_methods_group.common.ast.normalization.BasicASTNormalizer;
import org.ml_methods_group.common.serialization.ProtobufSerializationUtils;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class ParseAllNodes {
    public static void main(String[] args) throws IOException {
        final Dataset dataset = ProtobufSerializationUtils.loadDataset(
                Paths.get(".cache", "datasets", "train_dataset.tmp"));
        System.out.println("Loaded");
        final ASTGenerator generator = new BasicASTGenerator(new BasicASTNormalizer());
        int cnt = 0;
        final Path path = Paths.get(".cache", "out", "train_nodes.tmp");
        try (PrintWriter output = new PrintWriter(path.toFile())) {
            final List<Solution> values = dataset.getValues();
            for (Solution solution : values) {
                final ITree tree = generator.buildTree(solution);
                for (ITree node : tree.getTrees()) {
                    SerializationUtils.print(output, NodeState.getState(node));
                    SerializationUtils.print(output, NodeContext.getContext(node));
                    SerializationUtils.print(output, node.getDepth());
                    SerializationUtils.print(output, (double) node.getPos() / solution.getCode().length());
                    output.println("END");
                }
                cnt++;
                if (cnt % 100 == 0) {
                    System.out.println(100.0 * cnt / values.size());
                }
            }
        }
    }

    private static void printNodeContext(PrintWriter output, ITree node, String code) {
        printNode(output, node);
        printInt(output, node.positionInParent() == -1 ? 0 : node.positionInParent());
        printInt(output, node.getDepth());
        printFloat(output, (float) node.getPos() / code.length());

        final ITree parent = node.getParent();
        final ITree parentOfParent = parent == null ? null : parent.getParent();
        printNode(output, parent);
        printNode(output, parentOfParent);
        printNodesChildren(output, parentOfParent);
        printNodesChildren(output, parent);
        printNodesChildren(output, node);
    }

    private static void printNode(PrintWriter output, ITree node) {
        if (node == null) {
            printInt(output, -1);
            printLabel(output, "");
            printLabel(output, "");
        } else {
            printInt(output, node.getType());
            printLabel(output, getOriginalLabel(node));
            printLabel(output, getType(node));
        }
    }

    private static void printNodesChildren(PrintWriter output, ITree node) {
        if (node == null) {
            output.print(0);
            output.print(',');
            return;
        }
        final List<ITree> children = node.getChildren();
        output.print(children.size());
        output.print(',');
        for (ITree child : children) {
            printNode(output, child);
        }
    }

    private static void printLabel(PrintWriter output, String label) {
        output.print(label.replaceAll("[^a-zA-Z0-9@#~_%&*/\\-!=><()\\[\\]{}|^.?+]", ""));
        output.print(',');
    }

    private static void printInt(PrintWriter output, int value) {
        output.print(value);
        output.print(',');
    }

    private static void printFloat(PrintWriter output, float value) {
        output.print(Double.toString(value).replace(',', '.'));
        output.print(',');
    }

    private static String getType(ITree node) {
        final Object result = node.getMetadata(MetadataKeys.JAVA_TYPE);
        return result == null ? "" : result.toString();
    }

    private static String getOriginalLabel(ITree node) {
        final Object result = node.getMetadata(MetadataKeys.ORIGINAL_NAME);
        return result == null ? node.getLabel() : result.toString();
    }
}
