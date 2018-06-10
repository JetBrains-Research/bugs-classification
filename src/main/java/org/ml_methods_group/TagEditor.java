//package ru.spbau.mit.lobanov;
//
//import difflib.Delta;
//import difflib.DiffUtils;
//import ru.spbau.mit.lobanov.clusterization.Change;
//import ru.spbau.mit.lobanov.clusterization.ClusterizationResult;
//import ru.spbau.mit.lobanov.clusterization.Entity;
//import ru.spbau.mit.lobanov.database.Database;
//import ru.spbau.mit.lobanov.database.Table;
//
//import java.io.IOException;
//import java.io.UnsupportedEncodingException;
//import java.sql.SQLException;
//import java.util.*;
//import java.util.stream.Collectors;
//
//import static ru.spbau.mit.lobanov.database.Tables.codes_header;
//import static ru.spbau.mit.lobanov.database.Tables.tags_header;
//
//public class TagEditor {
//    private static final int problem = 53676;
//
//    public static void main(String[] args) throws Exception {
//        try (Database database = new Database();
//             Scanner input = new Scanner(System.in)) {
//            final Table codes = database.getTable(codes_header);
//            final Table tags = database.getTable(tags_header);
//            ClusterizationResult<Entity> clusterer = ReportViewer.createClusterer(database, problem);
//            List<List<Entity>> clusters = clusterer.clusters;
//            clusters.sort(Comparator.comparingInt(list -> -list.size()));
//            for (int i = 0; i < 70; i++) {
//                System.out.println("New cluster");
//
//                final HashSet<String> tagsList = new HashSet<>();
//                for (Entity entity : clusters.get(i)) {
//                    if (!printTags(entity.id + "", tags)) continue;
//                    if(addTags(entity.id + "", codes, tags, tagsList, input))
//                        break;
//                    System.out.println();
//                }
//            }
//        }
//    }
//
//    private static boolean printTags(String sessionId, Table tags) throws SQLException {
//        final Iterator<Table.ResultWrapper> iterator = tags.find("session_id", sessionId);
//        if (!iterator.hasNext()) return true;
//        System.out.print("Tags for " + sessionId + ": ");
//        while (iterator.hasNext()) {
//            final String tag = iterator.next().getStringValue("tag");
//            System.out.print(tag + " ");
//        }
//        System.out.println();
//        return false;
//    }
//
//    private static boolean addTags(String session, Table codes, Table tags,
//                               HashSet<String> tagsList, Scanner input) throws SQLException, IOException {
//        final String before = codes.findFirst(session + "_0")
//                .getStringValue("code");
//        final String after = codes.findFirst(session + "_1")
//                .getStringValue("code");
//        final List<Delta<String>> patch = DiffUtils.diff(lines(before), lines(after))
//                .getDeltas();
//        System.out.println("---before(" + session + ")---");
//        System.out.println(before);
//        System.out.println("-----------------------------");
//        System.out.println("---after(" + session + ")----");
//        System.out.println(after);
//        System.out.println("-----------------------------");
//        System.out.println("---diff(" + session + ")----");
//        patch.forEach(delta -> {
//            delta.getOriginal().getLines()
//                    .forEach(line -> System.out.println("-  " + line));
//            delta.getRevised().getLines()
//                    .forEach(line -> System.out.println("+  " + line));
//        });
//        System.out.println("-----------------------------");
//        System.out.println("AST diff: " + Change.getChanges(before, after)
//                .stream()
//                .map(Object::toString)
//                .collect(Collectors.joining()));
//        System.out.println("Used tags: " + tagsList.stream().collect(Collectors.joining(" ")));
//        while (true) {
//            System.out.print("tag: ");
//            final String tag = input.next();
//            if (tag.equals("-")) return false;
//            if (tag.equals("#")) return true;
//            System.out.println("Add " + tag + "?");
//            final String res = input.next();
//            if (res.equals("+")) {
//                tags.insert(new Object[]{session, tag});
//                System.out.println("Confirmed");
//                tagsList.add(tag);
//            } else {
//                System.out.println("Ignored");
//            }
//        }
//    }
//
//    private static List<String> lines(String text) {
//        final List<String> result = new ArrayList<>();
//        try (Scanner scanner = new Scanner(text)) {
//            while (scanner.hasNext()) {
//                result.add(scanner.nextLine());
//            }
//        }
//        return result;
//    }
//}
