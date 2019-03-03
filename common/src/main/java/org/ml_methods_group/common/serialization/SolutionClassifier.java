package org.ml_methods_group.common.serialization;

import org.ml_methods_group.common.Classifier;
import org.ml_methods_group.common.MarkedClusters;
import org.ml_methods_group.common.Solution;

import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;

public class SolutionClassifier implements Classifier<Solution, String>, Serializable {

    private final Classifier<Solution, String> classifier;

    public SolutionClassifier(Classifier<Solution, String> classifier) {
        this.classifier = classifier;
    }

    @Override
    public void train(MarkedClusters<Solution, String> samples) {
        classifier.train(samples);
    }

    @Override
    public Optional<String> classify(Solution value) {
        return classifier.classify(value);
    }

    @Override
    public Map<String, Double> reliability(Solution value) {
        return classifier.reliability(value);
    }

    @Override
    public Map.Entry<String, Double> mostProbable(Solution value) {
        return classifier.mostProbable(value);
    }

    public void store(Path path) throws IOException {
        SerializationUtils.storeObject(this, path);
    }

    public static SolutionClassifier load(Path path) throws IOException {
        return SerializationUtils.loadObject(SolutionClassifier.class, path);
    }
}
