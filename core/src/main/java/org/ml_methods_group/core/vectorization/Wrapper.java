package org.ml_methods_group.core.vectorization;

public class Wrapper {
    public final double[] vector;
    public final int sessionId;

    public Wrapper(double[] vector, int sessionId) {
        this.vector = vector;
        this.sessionId = sessionId;
    }

    public static double distance(Wrapper first, Wrapper second) {
        double result = 0;
        for (int i = 0; i < first.vector.length; i++) {
            result += (first.vector[i] - second.vector[i]) * (first.vector[i] - second.vector[i]);
        }
        return result;
    }
}
