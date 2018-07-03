package org.ml_methods_group.core.vectorization;

public class Wrapper {
    public final double[] vector;
    public final int sessionId;

    public Wrapper(double[] vector, int sessionId) {
        this.vector = vector;
        this.sessionId = sessionId;
    }

    public static double squaredEuclideanDistance(Wrapper first, Wrapper second) {
        double result = 0;
        for (int i = 0; i < first.vector.length; i++) {
            result += (first.vector[i] - second.vector[i]) * (first.vector[i] - second.vector[i]);
        }
        return result;
    }

    public static double euclideanDistance(Wrapper first, Wrapper second) {
        return Math.sqrt(squaredEuclideanDistance(first, second));
    }

    public static double manhattanDistance(Wrapper first, Wrapper second) {
        double result = 0;
        for (int i = 0; i < first.vector.length; i++) {
            result += Math.abs(first.vector[i] - second.vector[i]);
        }
        return result;
    }

    public double squaredEuclideanDistance(Wrapper other) {
        return squaredEuclideanDistance(this, other);
    }

    public double euclideanDistance(Wrapper other) {
        return euclideanDistance(this, other);
    }

    public double manhattanDistance(Wrapper other) {
        return manhattanDistance(this, other);
    }
}
