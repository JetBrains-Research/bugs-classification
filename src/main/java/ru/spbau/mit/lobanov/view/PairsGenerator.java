package ru.spbau.mit.lobanov.view;

import ru.spbau.mit.lobanov.clusterization.Wrap;
import ru.spbau.mit.lobanov.database.Database;
import ru.spbau.mit.lobanov.database.Table;
import ru.spbau.mit.lobanov.database.Tables;
import ru.spbau.mit.lobanov.preparation.DiffIndexer;
import ru.spbau.mit.lobanov.preparation.VectorTemplate;

import java.io.UnsupportedEncodingException;
import java.sql.SQLException;
import java.util.*;
import java.util.stream.Collectors;

public class PairsGenerator {

    private final Random random;
    private final Database database;
    private final VectorTemplate template;
    private final Scanner scanner;

    public PairsGenerator(long seed, Database database, VectorTemplate template) {
        this.random = new Random(seed);
        this.database = database;
        this.template = template;
        this.scanner = new Scanner(System.in);
    }

    public List<Pair> getInterestingPairs(int problem, int limit) throws Exception {
        final List<List<Wrap>> clusters = ViewUtils.clustersForProblem(database, template, problem).clusters;
        clusters.removeIf(list -> list.size() < 5);
        final List<Pair> result = new ArrayList<>();
        for (int i = 0; i < limit; i++) {
            result.add(new Pair(getRandomSample(clusters), getRandomSample(clusters)));
        }
        return result;
    }

    public List<Boolean> review(List<Pair> pairs) throws UnsupportedEncodingException, SQLException {
        final Table codes = database.getTable(Tables.codes_header);
        final List<Boolean> result = new ArrayList<>();
        for (Pair pair : pairs) {
            System.out.println();
            System.out.println();
            System.out.println();
            System.out.println("First sample:");
            ViewUtils.printDiff(codes, pair.first);
            System.out.println("Second sample:");
            ViewUtils.printDiff(codes, pair.second);
            System.out.println();
            System.out.println("Are they similar?");
            String response = scanner.next().toLowerCase();
            while (!response.equals("yes") && !response.equals("no")) {
                response = scanner.next().toLowerCase();
            }
            result.add(response.length() == 3);
        }
        return result;
    }

    public void runGeneration(int limit, int... problems) throws Exception {
        final Table pairs = database.getTable(Tables.train_pairs_header);
        final Object[] buffer = new Object[3];
        for (int problem : problems) {
            final List<Pair> generatedPairs = getInterestingPairs(problem, limit);
            final List<Boolean> isSimilar = review(generatedPairs);
            for (int i = 0; i < limit; i++) {
                buffer[0] = generatedPairs.get(i).first;
                buffer[1] = generatedPairs.get(i).second;
                buffer[2] = isSimilar.get(i);
                pairs.insert(buffer);
            }
        }
    }

    private int getRandomSample(List<List<Wrap>> clusters) {
        final int clusterIndex = random.nextInt(clusters.size());
        final int sessionIndex = random.nextInt(clusters.get(clusterIndex).size());
        return clusters.get(clusterIndex).get(sessionIndex).sessionId;
    }

    private class Pair {
        final int first;
        final int second;


        private Pair(int first, int second) {
            this.first = Math.min(first, second);
            this.second = Math.max(first, second);
        }
    }

    public static void main(String[] args) throws Exception {
        try (Database database = new Database()) {
            final Set<Long> types = DiffIndexer.getIndex(database)
                    .entrySet()
                    .stream()
                    .filter(e -> e.getValue() > 100)
                    .map(Map.Entry::getKey)
                    .collect(Collectors.toSet());
            final VectorTemplate template = new VectorTemplate(types, DiffIndexer.getDefaultStrategies());
            new PairsGenerator(239566, database, template).runGeneration(100,
                    55673, 100782, 27418, 113365);
//                    55673, 100782, 27418, 113365);
        }
    }
}
