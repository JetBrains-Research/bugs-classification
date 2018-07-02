package org.ml_methods_group.clusterization;

public class Wrap {
    public final double[] vector;
    public final int sessionId;

    public Wrap(double[] vector, int sessionId) {
        this.vector = vector;
        this.sessionId = sessionId;
    }

    public static double distance(Wrap first, Wrap second) {
        double result = 0;
        for (int i = 0; i < first.vector.length; i++) {
            result += (first.vector[i] - second.vector[i]) * (first.vector[i] - second.vector[i]);
        }
        return result;
    }
}
