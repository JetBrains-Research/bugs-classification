package org.ml_methods_group.view;

import difflib.DiffUtils;
import difflib.Patch;
import org.ml_methods_group.changes.AtomicChange;
import org.ml_methods_group.changes.ChangeUtils;
import org.ml_methods_group.clusterization.ClusterizationResult;
import org.ml_methods_group.clusterization.Wrap;
import org.ml_methods_group.database.Database;
import org.ml_methods_group.database.Tables;
import org.ml_methods_group.preparation.DiffIndexer;
import org.ml_methods_group.preparation.VectorTemplate;
import org.ml_methods_group.clusterization.HAC;
import org.ml_methods_group.database.Table;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.sql.SQLException;
import java.util.*;
import java.util.stream.Collectors;

public class ViewUtils {

    public static void printDiff(Table codes, int session) throws UnsupportedEncodingException, SQLException {
        final String before = codes.findFirst(session + "_0").getStringValue("code");
        final String after = codes.findFirst(session + "_1").getStringValue("code");
        final Patch<String> diff = DiffUtils.diff(lines(before), lines(after));
        System.out.println("--------------------------------------------");
        System.out.println("              Session id =" + session);
        System.out.println("--------------------------------------------");
        System.out.println(before);
        System.out.println("--------------------------------------------");
        System.out.println(after);
        System.out.println("--------------------------------------------");
        diff.getDeltas()
                .forEach(delta -> {
                    delta.getOriginal().getLines()
                            .stream()
                            .map(s -> "- " + s)
                            .forEachOrdered(System.out::println);
                    delta.getRevised().getLines()
                            .stream()
                            .map(s -> "+ " + s)
                            .forEachOrdered(System.out::println);
                });
        System.out.println("--------------------------------------------");
        System.out.println();
    }

    public static VectorTemplate defaultVectorTemplate(Database database) throws SQLException, FileNotFoundException {
        final Set<Long> types = DiffIndexer.getIndex(database)
                .entrySet()
                .stream()
                .filter(e -> e.getValue() > 100)
                .map(Map.Entry::getKey)
                .collect(Collectors.toSet());
        return new VectorTemplate(types, DiffIndexer.getDefaultStrategies());
    }

    public static List<Wrap> getSamples(Database database, VectorTemplate template, int problem) throws SQLException, IOException {
        final List<Wrap> samples = new ArrayList<>();
        final Table codes = database.getTable(Tables.codes_header);
        final Iterator<Table.ResultWrapper> iterator = codes.find("problem", problem);
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
            samples.add(new Wrap(template.toVector(changes), Integer.parseInt(sessionId)));
        }
        return samples;
    }

    public static List<Wrap> getRandomSamples(Database database, VectorTemplate template, int limit)
            throws SQLException, IOException {
        final List<Wrap> samples = new ArrayList<>();
        final Table codes = database.getTable(Tables.codes_header);
        final Iterator<Table.ResultWrapper> iterator = codes.listAll();
        final double probability = (double) limit / (codes.size() / 2);
        while (iterator.hasNext() && samples.size() < limit) {
            final Table.ResultWrapper result = iterator.next();
            final String id = result.getStringValue("id");
            if (id.endsWith("_0") || Math.random() < probability)
                continue;
            final String sessionId = id.substring(0, id.length() - 2);
            final String before = codes.findFirst(sessionId + "_0").getStringValue("code");
            final String after = codes.findFirst(sessionId + "_1").getStringValue("code");
            final List<AtomicChange> changes = ChangeUtils.calculateChanges(before, after);
            if (changes.size() == 0) continue;
            samples.add(new Wrap(template.toVector(changes), Integer.parseInt(sessionId)));
        }
        return samples;
    }

    public static ClusterizationResult<Wrap> clustersForProblem(Database database, VectorTemplate template,
                                                                int problemId) throws SQLException, IOException {
        final List<Wrap> train = getSamples(database, template, problemId);
        HAC<Wrap> hac = new HAC<>(train, Wrap::distance);
        final ClusterizationResult<Wrap> result = hac.run(0.12, train.size() / 10);
        hac.free();
        return result;
    }

    private static List<String> lines(String code) {
        return Arrays.asList(code.split("\n\r|\r|\n"));
    }
}
