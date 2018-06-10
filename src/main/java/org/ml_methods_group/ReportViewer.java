//package ru.spbau.mit.lobanov;
//
//import ru.spbau.mit.lobanov.clusterization.*;
//import ru.spbau.mit.lobanov.database.Database;
//import ru.spbau.mit.lobanov.database.Table;
//import ru.spbau.mit.lobanov.database.Tables;
//
//import java.sql.SQLException;
//import java.util.*;
//import java.util.stream.Collectors;
//import java.util.stream.IntStream;
//
//import static ru.spbau.mit.lobanov.database.Tables.tags_header;
//
//public class ReportViewer {
//    private static final int problem = 55673;
//
//    public static void main(String[] args) throws Exception {
//        try (Database database = new Database()) {
//            final Table tags = database.getTable(tags_header);
//            ClusterizationResult<Entity> clusters =
//                    createClusterer(database, problem);
//            printClusters(clusters, tags);
//        }
//    }
//
//    public static ClusterizationResult<Entity> createClusterer(Database database, int problemId) throws Exception {
//        final Table codes = database.getTable(Tables.codes_header);
//        List<Integer> train = new ArrayList<>();
//        final Iterator<Table.ResultWrapper> iterator = codes.find("problem", problemId);
//        while (iterator.hasNext()) {
//            final Table.ResultWrapper result = iterator.next();
//            final String id = result.getStringValue("id");
//            if (id.endsWith("_0"))
//                continue;
//            final String sessionId = id.substring(0, id.length() - 2);
//            final String before = codes.findFirst(sessionId + "_0").getStringValue("code");
//            final String after = codes.findFirst(sessionId + "_1").getStringValue("code");
//            final List<Change> changes = Change.getChanges(before, after);
//            if (changes.size() == 0) continue;
//            train.add(Integer.parseInt(sessionId));
//        }
//        HAC<Entity> hac = new HAC<>(Entity.loadEntity(train, database), new ChangeDistanceFunction());
//        return hac.run(0.5, 200);
//    }
//
//    public static void printClusters(ClusterizationResult<Entity> result, Table tags) throws Exception {
//        final List<Map<String, Integer>> counters = new ArrayList<>();
//        final Map<String, Integer> total = new HashMap<>();
//        final int[] sizeCounters = result.clusters.stream()
//                .mapToInt(List::size)
//                .toArray();
//        final List<List<String>> unknowns = new ArrayList<>();
//        final List<List<String>> clusters = new ArrayList<>();
//
//        for (List<Entity> cluster : result.clusters) {
//            final HashMap<String, Integer> counter = new HashMap<>();
//            final List<String> unknown = new ArrayList<>();
//            clusters.add(cluster.stream()
//                    .map(e -> e.id + "")
//                    .collect(Collectors.toList()));
//            for (Entity entity : cluster) {
//                if (getTags(counter, entity.id + "", tags))
//                    unknown.add(entity.id + "");
//            }
//            unknowns.add(unknown);
//            counters.add(counter);
//            if (cluster.size() <= 2) {
//                for (Map.Entry<String, Integer> e : counter.entrySet()) {
//                    total.put(e.getKey(), e.getValue() + total.getOrDefault(e.getKey(), 0));
//                }
//            }
//        }
//        final String r = total.entrySet()
//                .stream()
//                .sorted(Comparator.comparing(e -> -e.getValue()))
//                .limit(40)
//                .map(e -> e.getKey() + "(" + e.getValue() + ")")
//                .collect(Collectors.joining(", \n"));
//        IntStream.range(0, counters.size())
//                .boxed()
//                .sorted(Comparator.comparing(i -> -sizeCounters[i]))
//                .filter(i -> sizeCounters[i] >= 2)
//                .forEachOrdered(i -> {
//                    final String report = counters.get(i).entrySet()
//                            .stream()
//                            .sorted(Comparator.comparing(e -> -e.getValue()))
//                            .limit(10)
//                            .map(e -> e.getKey() + "(" + e.getValue() + ")")
//                            .collect(Collectors.joining(", "));
//                    System.out.println("Cluster #" + i + " [size = " + sizeCounters[i] +
//                            ", unknown = " + unknowns.get(i).size() + "] tags: " + report);
//                });
//        IntStream.range(0, counters.size())
//                .boxed()
//                .sorted(Comparator.comparing(i -> -sizeCounters[i]))
//                .forEachOrdered(i -> {
//                    System.out.println("Cluster #" + i);
//                    System.out.println("    all: " + clusters.get(i).stream().collect(Collectors.joining(", ")));
//                    System.out.println("    unknown: " + unknowns.get(i).stream().collect(Collectors.joining(", ")));
//                    System.out.println();
//                });
//    }
//
//
//    private static boolean getTags(Map<String, Integer> counters, String id, Table tags) throws SQLException {
//        final Iterator<Table.ResultWrapper> iterator = tags.find("session_id", id);
//        final boolean unknown = !iterator.hasNext();
//        while (iterator.hasNext()) {
//            final String tag = iterator.next().getStringValue("tag");
//            counters.put(tag, 1 + counters.getOrDefault(tag, 0));
//        }
//        return unknown;
//    }
//}
