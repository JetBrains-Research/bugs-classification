//package ru.spbau.mit.lobanov;
//
//import ru.spbau.mit.lobanov.clusterization.*;
//import ru.spbau.mit.lobanov.classification.Clusterer;
//import ru.spbau.mit.lobanov.classification.ClustersViewer;
//import ru.spbau.mit.lobanov.database.Database;
//import ru.spbau.mit.lobanov.database.Table;
//import ru.spbau.mit.lobanov.database.Tables;
//
//import java.util.*;
//
//public class Main {
//    private static final int problem = 55673;
//
//    public static void main(String[] args) throws Exception {
//        try (Database database = new Database()) {
//            ClusterizationResult<Entity> clusterizationResult = createClusterer(database, problem);
//            int cnt = 0;
//            for (List<Entity> cluster : clusterizationResult.clusters) {
//                System.out.println("cluster id " + (cnt++));
//                Collections.shuffle(cluster);
//                for (int j = 0; j < Math.min(10, cluster.size()); j++) {
//                    System.out.println();
//                    System.out.println();
//                    System.out.println("--------------before---------");
//                    System.out.println(diffs.get(j).before);
//                    System.out.println("--------------after---------");
//                    System.out.println(diffs.get(j).after);
//                    System.out.println("--------------diff---------");
//                    diffs.get(j).patches.forEach(System.out::println);
//                }
//            }
//        }
//    }
//
//    private static ClusterizationResult<Entity> createClusterer(Database database, int problemId) throws Exception {
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
//            final List<AtomicChange> changes = AtomicChange.getChanges(before, after);
//            if (changes.size() == 0) continue;
//            train.add(Integer.parseInt(sessionId));
//        }
//        HAC<Entity> hac = new HAC<>(Entity.loadEntity(train, database), new ChangeDistanceFunction());
//        return hac.run(0.5, 200);
//    }
//}
