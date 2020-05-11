package org.ml_methods_group.evaluation.approaches.classification;

import org.ml_methods_group.classification.classifiers.CompositeClassifier;
import org.ml_methods_group.classification.classifiers.KNearestNeighbors;
import org.ml_methods_group.classification.classifiers.NearestCluster;
import org.ml_methods_group.common.Classifier;
import org.ml_methods_group.common.CommonUtils;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.Wrapper;
import org.ml_methods_group.evaluation.approaches.Approach;

import java.util.function.Function;

public class ClassificationApproach {

    private final Function<String, Classifier<Solution, String>> creator;
    private final String name;

    public <T> ClassificationApproach(Approach<T> approach) {
        this.creator = name -> new CompositeClassifier<>(
                approach.extractor,
                classifierForName(name, approach));
        this.name = approach.name;
    }

    public Classifier<Solution, String> getClassifier(String classifierName) {
        return creator.apply(classifierName);
    }

    public String getName() {
        return name;
    }

    private static <T> Classifier<Wrapper<T, Solution>, String> classifierForName(String name, Approach<T> approach) {
        final var metric = CommonUtils.metricFor(approach.metric, Wrapper<T, Solution>::getFeatures);
        switch (name) {
            case "closest-cluster":
                return new NearestCluster<>(metric);
            case "k-nearest-3":
                return new KNearestNeighbors<>(3, metric);
            case "k-nearest-5":
                return new KNearestNeighbors<>(5, metric);
            case "k-nearest-10":
                return new KNearestNeighbors<>(10, metric);
            case "k-nearest-15":
                return new KNearestNeighbors<>(15, metric);
            case "k-nearest-20":
                return new KNearestNeighbors<>(20, metric);
            default:
                throw new IllegalArgumentException();
        }
    }
}
