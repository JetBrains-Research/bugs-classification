package org.ml_methods_group.core;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public class Scenario<V, M> {
    private final Clusterer<V> clusterer;
    private final Classifier<V, M> classifier;
    private final Marker<Cluster<V>, M> marker;
    private final List<V> samples;

    public Scenario(Clusterer<V> clusterer, Classifier<V, M> classifier,
                    Marker<Cluster<V>, M> marker, List<V> samples) {
        this.clusterer = clusterer;
        this.classifier = classifier;
        this.marker = marker;
        this.samples = samples;
    }

    public void run() {
        final List<Cluster<V>> clusters = clusterer.buildClusters(samples);
        //todo
        System.out.println(clusters.size());
        System.out.println(samples.size());
        System.out.println(clusters.stream().filter(l -> l.size() >= 5).count());
        System.out.println(clusters.stream().filter(l -> l.size() >= 5).mapToInt(Cluster::size).sum());
        System.out.println(clusters.stream().filter(l -> l.size() >= 10).count());
        System.out.println(clusters.stream().filter(l -> l.size() >= 10).mapToInt(Cluster::size).sum());
        System.out.println(clusters.stream().filter(l -> l.size() >= 15).count());
        System.out.println(clusters.stream().filter(l -> l.size() >= 15).mapToInt(Cluster::size).sum());
        System.out.println(clusters.stream().filter(l -> l.size() >= 20).count());
        System.out.println(clusters.stream().filter(l -> l.size() >= 20).mapToInt(Cluster::size).sum());
        final Map<Cluster<V>, M> marks = clusters.stream()
                .collect(Collectors.toMap(Function.identity(), marker::mark));
        classifier.train(marks);
    }

    public Classifier<V, M> getClassifier() {
        return classifier;
    }
}
