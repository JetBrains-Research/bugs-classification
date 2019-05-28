package org.ml_methods_group.evaluation.vectorization;

import org.ml_methods_group.common.ast.changes.CodeChange;
import org.ml_methods_group.common.ast.changes.CodeChange.NodeContext;
import org.ml_methods_group.common.ast.changes.CodeChange.NodeState;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Scanner;

public class SerializationUtils {
    public static void print(PrintWriter output, int value) {
        output.print(value);
        output.print(',');
    }

    public static void print(PrintWriter output, String label) {
        output.print(label.replaceAll("[^a-zA-Z0-9@#~_%&*/\\-!=><()\\[\\]{}|^.?+]", ""));
        output.print(',');
    }

    public static void print(PrintWriter output, double value) {
        output.print(Double.toString(value).replace(',', '.'));
        output.print(',');
    }

    public static void print(PrintWriter output, NodeState state) {
        print(output, state.getType().ordinal());
        print(output, state.getPositionInParent());
        print(output, state.getLabel());
        print(output, state.getOriginalLabel());
        print(output, state.getJavaType());
    }

    public static void print(PrintWriter output, NodeState[] states) {
        print(output, states.length);
        for (NodeState state : states) {
            print(output, state);
        }
    }

    public static void print(PrintWriter output, NodeContext context) {
        print(output, context.getNode());
        print(output, context.getParent());
        print(output, context.getParentOfParent());
        print(output, context.getChildren());
        print(output, context.getBrothers());
        print(output, context.getUncles());
    }

    public static void print(PrintWriter output, CodeChange change) {
        print(output, change.getChangeType().ordinal());
        print(output, change.getOriginalContext());
        print(output, change.getDestinationContext());
    }

    public static Map<String, Integer> readMap(String filename) throws FileNotFoundException {
        final HashMap<String, Integer> result = new HashMap<>();
        try (Scanner scanner = new Scanner(new File(filename))) {
            final int id = scanner.nextInt();
            final String token = scanner.nextLine().trim();
            result.put(token, id);
        }
        return result;
    }

    public static Map<Integer, double[]> readEmbedding(String filename) throws FileNotFoundException {
        final HashMap<Integer, double[]> result = new HashMap<>();
        try (Scanner scanner = new Scanner(new File(filename))) {
            scanner.useLocale(Locale.US);
            final int size = scanner.nextInt();
            while (scanner.hasNext()) {
                final int id = scanner.nextInt();
                final double[] vector = new double[size];
                for (int i = 0; i < size; i++) {
                    vector[i] = scanner.nextDouble();
                }
                result.put(id, vector);
            }
        }
        return result;
    }
}
