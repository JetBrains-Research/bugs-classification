package ru.spbau.mit.lobanov;

import ru.spbau.mit.lobanov.changes.AtomicChange;
import ru.spbau.mit.lobanov.changes.ChangeUtils;
import ru.spbau.mit.lobanov.clusterization.ClusterizationResult;
import ru.spbau.mit.lobanov.clusterization.HAC;
import ru.spbau.mit.lobanov.clusterization.Wrap;
import ru.spbau.mit.lobanov.database.Database;
import ru.spbau.mit.lobanov.database.Table;
import ru.spbau.mit.lobanov.database.Tables;
import ru.spbau.mit.lobanov.preparation.DiffIndexer;
import ru.spbau.mit.lobanov.preparation.VectorTemplate;
import ru.spbau.mit.lobanov.view.ViewUtils;

import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.sql.SQLException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class HAC2Try {

    //    static final int problem = 55673;
    static final int problem = 58088;

    public static void main(String[] args) throws Exception {
        final Map<Long, Integer> localIndex = FileUtils.readDictionary("index2.txt", Long::parseLong, Integer::parseInt);
        final VectorTemplate template = new VectorTemplate(localIndex.entrySet().stream()
                .filter(e -> e.getValue() > 2)
                .map(java.util.Map.Entry::getKey)
                .collect(Collectors.toSet()),
                DiffIndexer.getDefaultStrategies());
        Database database = new Database();
        List<Wrap> samples = ViewUtils.getSamples(database, template, problem);
        System.out.println(samples.size());
        Collections.shuffle(samples, new Random(1234));

        HAC<Wrap> hac = new HAC<>(samples.subList(20, samples.size()), Wrap::distance);
        final ClusterizationResult<Wrap> result = hac.run(0.3, 80);
        System.out.println(result.clusters.size());
        System.out.println(result.clusters.stream()
                .mapToInt(List::size)
                .filter(i -> i > 4)
                .count());
        System.out.println(result.clusters.stream()
                .mapToInt(List::size)
                .filter(i -> i > 4)
                .sum());
        hac.free();
//        printClusters(result, database.getTable(Tables.tags_header));
        testClusters(result, database.getTable(Tables.codes_header), samples.subList(0, 20),
                samples.subList(20, samples.size()), template);
        database.close();
    }


    public static void testClusters(ClusterizationResult<Wrap> result, Table codes,
                                    List<Wrap> test, List<Wrap> others, VectorTemplate template) throws Exception {
        Scanner scanner = new Scanner(System.in);
        final List<Map<String, Integer>> counters = new ArrayList<>();
        final Map<String, Integer> total = new HashMap<>();
        final int[] sizeCounters = result.clusters.stream()
                .mapToInt(List::size)
                .sorted()
                .toArray();
        PrintWriter out = new PrintWriter("output.csv");
        int sum = 0;
        int cnt = 0;
        for (int i = sizeCounters.length - 1; i >= 0; i--) {
            sum += sizeCounters[i];
            cnt++;
            out.write(cnt + "," + sum + "\n");
        }
        out.close();
        final int all = result.clusters.stream().mapToInt(List::size).sum();
        int viewed = 0;
        System.out.println(Arrays.toString(sizeCounters));
        for (Wrap t : test) {
            ViewUtils.printDiff(codes, t.sessionId);
            final String code = codes.findFirst(t.sessionId + "_0").getStringValue("code");
            List<AtomicChange> changes = changes(code, others, codes);
            System.out.println("Changes:" + changes);
            double[] vec = template.toVector(changes);
            System.out.println(Arrays.toString(vec));
            double bestDist = Double.POSITIVE_INFINITY;
            List<Wrap> bestCluster = null;
            for (List<Wrap> cluster : result.clusters) {
                double dist = 0;
                if (cluster.size() < 5) continue;
                for (Wrap w : cluster) {
                    dist += result.distanceFunction.distance(w, new Wrap(vec, -1));
                }
                dist /= cluster.size();
                if (dist < bestDist) {
                    bestDist = dist;
                    bestCluster = cluster;
                }
            }
            Collections.shuffle(bestCluster);
            for (int i = 0; i < 5; i++) {
                ViewUtils.printDiff(codes, bestCluster.get(i).sessionId);
            }
        }
//        result.clusters.sort(Comparator.comparingInt(list -> -list.size()));
//
//        for (List<Wrap> cluster : result.clusters) {
//            viewed += cluster.size();
//            System.out.println("Next cluster: " + cluster.size());
//            System.out.println("Total: " + ((double) viewed / all));
//            Collections.shuffle(cluster);
//            for (Wrap wrap : cluster) {
//                ViewUtils.printDiff(codes, wrap.sessionId);
//                String s = scanner.nextLine();
//                if (s.equals("next"))
//                    break;
//            }
//        }
    }

    private static List<AtomicChange> changes(String code, List<Wrap> train, Table codes) throws IOException, SQLException {
        int diffs = 1000000;
        int bestSession = -1;
        for (Wrap wrap : train) {
            String after = codes.findFirst(wrap.sessionId + "_1").getStringValue("code");
            final List<AtomicChange> changes = ChangeUtils.calculateChanges(code, after);
            if (diffs > changes.size() && changes.size() > 0) {
                diffs = changes.size();
                bestSession = wrap.sessionId;
            }
        }
        String after = codes.findFirst(bestSession + "_1").getStringValue("code");
        return ChangeUtils.calculateChanges(code, after);
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
}
