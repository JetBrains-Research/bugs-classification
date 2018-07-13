package org.ml_methods_group.core.vectorization;

import org.ml_methods_group.core.FeaturesExtractor;
import org.ml_methods_group.core.entities.CodeChange;

public class Vectorizer implements FeaturesExtractor<CodeChange, double[]> {
    @Override
    public double[] process(CodeChange value) {
        return new double[0];
    }
}
