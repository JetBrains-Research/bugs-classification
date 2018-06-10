package ru.spbau.mit.lobanov;

import Jama.Matrix;
import com.mkobos.pca_transform.PCA;
import ru.spbau.mit.lobanov.clusterization.ClusterizationResult;
import ru.spbau.mit.lobanov.clusterization.HAC;
import ru.spbau.mit.lobanov.clusterization.Wrap;
import ru.spbau.mit.lobanov.database.Database;
import ru.spbau.mit.lobanov.database.Table;
import ru.spbau.mit.lobanov.database.Tables;
import ru.spbau.mit.lobanov.view.ViewUtils;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.sql.SQLException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static ru.spbau.mit.lobanov.view.ViewUtils.defaultVectorTemplate;


public class SandboxPCA {

    static final int problem = 55673;

    public static void main(String[] args) throws Exception {
        Database database = new Database();
        List<Wrap> samples = ViewUtils.getRandomSamples(database, defaultVectorTemplate(database), 10000);
        double[][] data = new double[samples.size()][];
        for (int i = 0; i < samples.size(); i++) {
            data[i] = samples.get(i).vector;
        }
        Matrix trainingData = new Matrix(data);
        PCA pca = new PCA(trainingData);
        Matrix transformedData =
                pca.transform(trainingData, PCA.TransformationType.WHITENING);
        System.out.println("Transformed data (each row corresponding to transformed data point):");
        System.out.println(transformedData.getRowDimension());
        System.out.println(transformedData.getColumnDimension());
        double[][] fixedData = new double[trainingData.getRowDimension()][trainingData.getColumnDimension()];
        for (int i = 0; i < fixedData.length; i++) {
            for (int j = 0; j < fixedData[0].length; j++) {
                fixedData[i][j] = trainingData.get(i, j);
            }
        }
        List<Wrap> fixedSamples = new ArrayList<>();
        for (int i = 0; i < samples.size(); i++) {
            fixedSamples.add(new Wrap(fixedData[i], samples.get(i).sessionId));
        }
        HAC<Wrap> hac = new HAC<>(fixedSamples, Wrap::distance);
        final ClusterizationResult<Wrap> result = hac.run(0.1, 200);
        printClusters(result, database.getTable(Tables.tags_header));
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
