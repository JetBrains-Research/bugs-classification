//package ru.spbau.mit.lobanov;
//
//import ru.spbau.mit.lobanov.clusterization.AtomicChange;
//import ru.spbau.mit.lobanov.classification.Clusterer;
//import ru.spbau.mit.lobanov.classification.ClustersViewer;
//import ru.spbau.mit.lobanov.database.Database;
//import ru.spbau.mit.lobanov.database.Table;
//
//import java.io.FileOutputStream;
//import java.io.ObjectOutputStream;
//import java.sql.SQLException;
//import java.util.ArrayList;
//import java.util.Iterator;
//import java.util.List;
//
//import static ru.spbau.mit.lobanov.database.Tables.codes_header;
//import static ru.spbau.mit.lobanov.database.Tables.tags_header;
//
//public class Sandbox {
//    static final int problem = 55673;
//
//    public static void main(String[] args) throws Exception {
//        try (Database database = new Database();
//            FileOutputStream fos = new FileOutputStream("clusters_55673.oos");
//            ObjectOutputStream oos = new ObjectOutputStream(fos)) {
//            final Table tags = database.getTable(tags_header);
//            Clusterer clusterer = createClusterer(database, problem);
//            ClustersViewer viewer = new ClustersViewer(clusterer, problem, database);
//            oos.writeInt(clusterer.getImplementation().getNumClusters());
//            for (int i = 0; i < clusterer.getImplementation().getNumClusters(); i++) {
//                final List<ClustersViewer.Diff> diffs = viewer.getCluster(i);
//                final List<List<String>> tagsLists = new ArrayList<>();
//                for (ClustersViewer.Diff diff : diffs) {
//                    tagsLists.add(printTags(diff.sessionId, tags));
//                }
//                oos.writeObject(diffs);
//                oos.writeObject(tagsLists);
//                System.out.println(100.0 * i / clusterer.getImplementation().getNumClusters());
//            }
//        }
//    }
//
//    private static List<String> printTags(String sessionId, Table tags) throws SQLException {
//        final Iterator<Table.ResultWrapper> iterator = tags.find("session_id", sessionId);
//        List<String> res = new ArrayList<>();
//        while (iterator.hasNext()) {
//            final String tag = iterator.next().getStringValue("tag");
//            res.add(tag);
//        }
//        return res;
//    }
//
//    private static Clusterer createClusterer(Database database, int problemId) throws Exception {
//        final Table codes = database.getTable(codes_header);
//        List<List<AtomicChange>> train = new ArrayList<>();
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
//            train.add(changes);
//        }
//        Clusterer clusterer = new Clusterer();
//        clusterer.build(train);
//        Utils.saveClusterer("clusterer.bin", clusterer);
//        return clusterer;
//    }
//}
