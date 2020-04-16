package org.ml_methods_group.common.metrics.functions;

import com.google.common.collect.Sets;

import java.util.Arrays;
import java.util.Map;
import java.util.Set;

public class FunctionsUtils {
    public static int scalarProduct(int[] a, int[] b) {
        int sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    public static int scalarProduct(Map<Integer, Integer> countersA, Map<Integer, Integer> countersB) {
        Set<Integer> intersection = Sets.intersection(countersA.keySet(), countersB.keySet());
        return intersection.stream()
                .mapToInt(x -> countersA.get(x) * countersB.get(x))
                .sum();
    }

    public static double cosineSimilarity(double[] a, double[] b) {
        double sum = 0;
        double normA = 0;
        double normB = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return sum == 0 ? 0 : (1 + sum / Math.sqrt(normA * normB)) / 2;
    }

    public static double[] sum(double[]... vectors) {
        final double[] result = new double[vectors[0].length];
        for (double[] vector : vectors) {
            for (int i = 0; i < result.length; i++) {
                result[i] += vector[i];
            }
        }
        return result;
    }


    public static double[] sum(double[] a, double[] b) {
        final double[] result = new double[a.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = a[i] + b[i];
        }
        return result;
    }


    public static void add(double[] vector, double[] delta) {
        for (int i = 0; i < vector.length; i++) {
            vector[i] += delta[i];
        }
    }


    public static double[] scale(double[] vector, double scale) {
        final double[] result = new double[vector.length];
        for (int i = 0; i < vector.length; i++) {
            result[i] = vector[i] * scale;
        }
        return result;
    }

    public static double cosineDistance(double[] a, double[] b) {
        return 1 - cosineSimilarity(a, b);
    }

    public static double norm(double[] a) {
        return Math.sqrt(Arrays.stream(a).map(x -> x * x).sum());
    }
}
