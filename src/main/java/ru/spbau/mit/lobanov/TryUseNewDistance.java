package ru.spbau.mit.lobanov;

import org.encog.ml.factory.method.SOMFactory;
import org.encog.neural.som.SOM;
import ru.spbau.mit.lobanov.changes.AtomicChange;
import ru.spbau.mit.lobanov.changes.ChangeUtils;
import ru.spbau.mit.lobanov.clusterization.ClusterizationResult;
import ru.spbau.mit.lobanov.clusterization.HAC;
import ru.spbau.mit.lobanov.database.Database;
import ru.spbau.mit.lobanov.database.Table;
import ru.spbau.mit.lobanov.database.Tables;
import ru.spbau.mit.lobanov.preparation.DiffIndexer;
import ru.spbau.mit.lobanov.preparation.VectorTemplate;


import java.sql.SQLException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class TryUseNewDistance {

    private static final int problem = 55673;

    public static void main(String[] args) throws Exception {
        try (Database database = new Database()) {
            final Set<Long> types = DiffIndexer.getIndex(database)
                    .entrySet()
                    .stream()
                    .filter(e -> e.getValue() > 100)
                    .map(Map.Entry::getKey)
                    .collect(Collectors.toSet());
            final VectorTemplate template = new VectorTemplate(types, DiffIndexer.getDefaultStrategies());
            ClusterizationResult<Wrap> clusterizationResult = createClusterer(database, template, problem);
            printClusters(clusterizationResult, database.getTable(Tables.tags_header));
        }
    }

    static class Wrap {
        public final double[] vector;
        public final String sessionId;

        Wrap(double[] vector, String sessionId) {
            this.vector = vector;
            this.sessionId = sessionId;
        }
    }

    private static ClusterizationResult<Wrap> createClusterer2(Database database, VectorTemplate converter, int problemId) throws Exception {
        final Table codes = database.getTable(Tables.codes_header);
        List<Wrap> train = new ArrayList<>();
        final Iterator<Table.ResultWrapper> iterator = codes.find("problem", problemId);
        while (iterator.hasNext()) {
            final Table.ResultWrapper result = iterator.next();
            final String id = result.getStringValue("id");
            if (id.endsWith("_0"))
                continue;
            final String sessionId = id.substring(0, id.length() - 2);
            final String before = codes.findFirst(sessionId + "_0").getStringValue("code");
            final String after = codes.findFirst(sessionId + "_1").getStringValue("code");
            final List<AtomicChange> changes = ChangeUtils.calculateChanges(before, after);
            if (changes.size() == 0) continue;
            train.add(new Wrap(converter.toVector(changes), sessionId));
        }
        final SOM som = new SOM(100, 100);
        System.out.println("Run HAC...");
        HAC<Wrap> hac = new HAC<>(train, TryUseNewDistance::distance);
        final ClusterizationResult<Wrap> result = hac.run(0.4, 100);
        hac.free();
        return result;
    }

    private static ClusterizationResult<Wrap> createClusterer(Database database, VectorTemplate converter, int problemId) throws Exception {
        final Table codes = database.getTable(Tables.codes_header);
        List<Wrap> train = new ArrayList<>();
        final Iterator<Table.ResultWrapper> iterator = codes.find("problem", problemId);
        while (iterator.hasNext()) {
            final Table.ResultWrapper result = iterator.next();
            final String id = result.getStringValue("id");
            if (id.endsWith("_0"))
                continue;
            final String sessionId = id.substring(0, id.length() - 2);
            final String before = codes.findFirst(sessionId + "_0").getStringValue("code");
            final String after = codes.findFirst(sessionId + "_1").getStringValue("code");
            final List<AtomicChange> changes = ChangeUtils.calculateChanges(before, after);
            if (changes.size() == 0) continue;
            train.add(new Wrap(converter.toVector(changes), sessionId));
        }
        System.out.println("Run HAC...");
        HAC<Wrap> hac = new HAC<>(train, TryUseNewDistance::distance);
        final ClusterizationResult<Wrap> result = hac.run(0.4, 100);
        hac.free();
        return result;
    }

    public static double distance(Wrap a, Wrap b) {
        double result = 0;
        for (int i = 0; i < a.vector.length; i++) {
            result += (a.vector[i] - b.vector[i]) * (a.vector[i] - b.vector[i]);
        }
        return result;
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
                if (getTags(counter, entity.sessionId, tags))
                    unknown.add(entity.sessionId);
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
}
