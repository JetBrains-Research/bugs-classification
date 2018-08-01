package org.ml_methods_group.core;

import org.ml_methods_group.core.entities.Solution;

import java.io.*;
import java.util.Map;

public interface FeaturesExtractor<F> extends Serializable {
    F process(Solution value, Solution target);
    void train(Map<Solution, Solution> dataset);
}
