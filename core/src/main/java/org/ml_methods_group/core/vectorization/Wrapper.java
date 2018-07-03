package org.ml_methods_group.core.vectorization;

import java.io.Serializable;

public class Wrapper implements Serializable {
    public final double[] vector;
    public final int sessionId;

    public Wrapper(double[] vector, int sessionId) {
        this.vector = vector;
        this.sessionId = sessionId;
    }

    public double squaredEuclideanDistance(Wrapper other) {
        return Utils.squaredEuclideanDistance(vector, other.vector);
    }

    public double euclideanDistance(Wrapper other) {
        return Utils.euclideanDistance(vector, other.vector);
    }

    public double manhattanDistance(Wrapper other) {
        return Utils.manhattanDistance(vector, other.vector);
    }
}
