package org.ml_methods_group.core;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

public class Scenario<V, M> {
    private final Clusterer<V> clusterer;
    private final Classifier<V, M> classifier;
    private final Marker<List<V>, M> marker;
    private final Predicate<List<V>> filter;
    private final List<V> samples;

    public Scenario(Clusterer<V> clusterer, Classifier<V, M> classifier,
                    Marker<List<V>, M> marker, Predicate<List<V>> filter, List<V> samples) {
        this.clusterer = clusterer;
        this.classifier = classifier;
        this.marker = marker;
        this.filter = filter;
        this.samples = samples;
    }

    public void run() {
        final List<List<V>> clusters = clusterer.buildClusters(samples);
        final Map<V, M> marks = new HashMap<>();
        for (List<V> cluster : clusters) {
            if (!filter.test(cluster)) {
                continue;
            }
            final M mark = marker.mark(cluster);
            cluster.forEach(e -> marks.put(e, mark));
        }
        classifier.train(marks);
    }

    public Classifier<V, M> getClassifier() {
        return classifier;
    }
}
