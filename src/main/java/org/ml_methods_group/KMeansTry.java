package org.ml_methods_group;

import org.ml_methods_group.preparation.DiffIndexer;
import org.ml_methods_group.preparation.VectorTemplate;

import java.io.FileNotFoundException;
import java.sql.SQLException;
import java.util.Map;
import java.util.stream.Collectors;

public class KMeansTry {

    static final int problem = 55673;

    public static void main(String[] args) throws SQLException, FileNotFoundException {
        final Map<Long, Integer> localIndex = FileUtils.readDictionary("local_index", Long::parseLong, Integer::parseInt);
        final VectorTemplate template = new VectorTemplate(localIndex.entrySet().stream()
                .filter(e -> e.getValue() > 1)
                .map(Map.Entry::getKey)
                .collect(Collectors.toSet()),
                DiffIndexer.getDefaultStrategies());

//        Database database = new Database();

//        FileUtils.writeDictionary("local_index", localIndex);
        System.out.println(localIndex.size());
        System.out.println(localIndex.values().stream().filter(i -> i > 1).count());
        System.out.println(localIndex.values().stream().filter(i -> i > 5).count());
        System.out.println(localIndex.values().stream().filter(i -> i > 2).count());
        System.out.println(localIndex.values().stream().filter(i -> i > 10).count());
//        database.close();
    }
}
