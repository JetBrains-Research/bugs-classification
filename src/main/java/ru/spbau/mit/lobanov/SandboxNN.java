package ru.spbau.mit.lobanov;

import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.som.SOM;
import org.encog.neural.som.training.basic.BasicTrainSOM;
import org.encog.neural.som.training.basic.neighborhood.NeighborhoodSingle;
import ru.spbau.mit.lobanov.clusterization.ClusterizationResult;
import ru.spbau.mit.lobanov.clusterization.Wrap;
import ru.spbau.mit.lobanov.database.Database;
import ru.spbau.mit.lobanov.database.Table;
import ru.spbau.mit.lobanov.database.Tables;
import ru.spbau.mit.lobanov.preparation.VectorTemplate;
import ru.spbau.mit.lobanov.view.ViewUtils;
import weka.clusterers.SimpleKMeans;

import java.sql.SQLException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


public class SandboxNN {

    static final int problem = 55673;

    public static void main(String[] args) throws Exception {
        try (Database database = new Database()) {
            final VectorTemplate template = ViewUtils.defaultVectorTemplate(database);
            final List<Wrap> samples = ViewUtils.getSamples(database, template, problem);
            SOM som = new SOM(template.size(), 100);
            som.reset();
            SimpleKMeans sk = new SimpleKMeans();
            final BasicTrainSOM train = new BasicTrainSOM(som, 0.4, toDataset(samples, template),
                    new NeighborhoodSingle());
            train.setForceWinner(true);
            for (int i = 0; i < 2000; i++) {
                System.out.println(i);
                train.iteration();
            }
            final List<List<Wrap>> result = samples.stream()
                    .collect(Collectors.groupingBy(w -> som.classify(new BasicMLData(w.vector)), Collectors.toList()))
                    .entrySet()
                    .stream()
                    .map(Map.Entry::getValue)
                    .collect(Collectors.toList());
            printClusters(new ClusterizationResult<>(result, null), database.getTable(Tables.tags_header));
        }
    }

    static MLDataSet toDataset(List<Wrap> wraps, VectorTemplate template) {
        double[][] set = new double[wraps.size()][];
        for (int i = 0; i < set.length; i++) {
            set[i] = wraps.get(i).vector;
        }
        return new BasicMLDataSet(set, null);
    }

    public static void printClusters(ClusterizationResult<Wrap> result, Table tags) throws Exception {
        final List<Map<String, Integer>> counters = new ArrayList<>();
        final Map<String, Integer> total = new HashMap<>();
        final int[] sizeCounters = result.clusters.stream()
                .mapToInt(List::size)
                .toArray();
        final List<List<String>> unknowns = new ArrayList<>();
        final List<List<String>> clusters = new ArrayList<>();

        for (List<Wrap> cluster : result.clusters) {
            final HashMap<String, Integer> counter = new HashMap<>();
            final List<String> unknown = new ArrayList<>();
            clusters.add(cluster.stream()
                    .map(e -> e.sessionId + "")
                    .collect(Collectors.toList()));
            for (Wrap entity : cluster) {
                if (getTags(counter, entity.sessionId + "", tags))
                    unknown.add(entity.sessionId + "");
            }
            unknowns.add(unknown);
            counters.add(counter);
            if (cluster.size() <= 2) {
                for (Map.Entry<String, Integer> e : counter.entrySet()) {
                    total.put(e.getKey(), e.getValue() + total.getOrDefault(e.getKey(), 0));
                }
            }
        }
        final String r = total.entrySet()
                .stream()
                .sorted(Comparator.comparing(e -> -e.getValue()))
                .limit(40)
                .map(e -> e.getKey() + "(" + e.getValue() + ")")
                .collect(Collectors.joining(", \n"));
        IntStream.range(0, counters.size())
                .boxed()
                .sorted(Comparator.comparing(i -> -sizeCounters[i]))
                .filter(i -> sizeCounters[i] >= 2)
                .forEachOrdered(i -> {
                    final String report = counters.get(i).entrySet()
                            .stream()
                            .sorted(Comparator.comparing(e -> -e.getValue()))
                            .limit(10)
                            .map(e -> e.getKey() + "(" + e.getValue() + ")")
                            .collect(Collectors.joining(", "));
                    System.out.println("Cluster #" + i + " [size = " + sizeCounters[i] +
                            ", unknown = " + unknowns.get(i).size() + "] tags: " + report);
                });
        IntStream.range(0, counters.size())
                .boxed()
                .sorted(Comparator.comparing(i -> -sizeCounters[i]))
                .forEachOrdered(i -> {
                    System.out.println("Cluster #" + i);
                    System.out.println("    all: " + clusters.get(i).stream().collect(Collectors.joining(", ")));
                    System.out.println("    unknown: " + unknowns.get(i).stream().collect(Collectors.joining(", ")));
                    System.out.println();
                });
    }

    private static boolean getTags(Map<String, Integer> counters, String id, Table tags) throws SQLException {
        final Iterator<Table.ResultWrapper> iterator = tags.find("session_id", id);
        final boolean unknown = !iterator.hasNext();
        while (iterator.hasNext()) {
            final String tag = iterator.next().getStringValue("tag");
            counters.put(tag, 1 + counters.getOrDefault(tag, 0));
        }
        return unknown;
    }


    private static void incrementCounter(Map<Long, Integer> index, long type) {
        index.compute(type, (k, cnt) -> cnt == null ? 1 : cnt + 1);
    }
}
