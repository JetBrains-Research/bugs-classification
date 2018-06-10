package ru.spbau.mit.lobanov.clusterization;

import java.io.Serializable;

public interface DistanceFunction<T> extends Serializable {
    double distance(T first, T second);
}
