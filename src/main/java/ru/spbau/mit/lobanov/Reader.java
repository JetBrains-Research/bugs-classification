//package ru.spbau.mit.lobanov;
//
//import java.io.FileInputStream;
//import java.io.IOException;
//import java.io.ObjectInputStream;
//import java.util.List;
//import java.util.Scanner;
//import java.util.stream.Collectors;
//
//import static ru.spbau.mit.lobanov.classification.ClustersViewer.Diff;
//
//public class Reader {
//    public static void main(String[] args) throws IOException, ClassNotFoundException {
//        boolean showBefore = args[1].contains("b");
//        boolean showAfter = args[1].contains("a");
//        boolean showDiff = args[1].contains("d");
//        try (FileInputStream fis = new FileInputStream(args[0]);
//             ObjectInputStream ois = new ObjectInputStream(fis);
//             Scanner scanner = new Scanner(System.in)) {
//            int n = ois.readInt();
//            for (int i = 0; i < n; i++) {
//                List<Diff> difs = (List<Diff>) ois.readObject();
//                List<List<String>> tags = (List<List<String>>) ois.readObject();
//                System.out.println("In cluster #" + i + " " + difs.size() + " items");
//                System.out.println("Insert s to skip cluster, n to show first solution");
//                String command = scanner.next();
//                while (!command.equals("s") && !command.equals("n")) {
//                    System.out.println("Unrecognized command");
//                    System.out.println("Insert s to skip cluster, n to show first solution");
//                    command = scanner.next();
//                }
//                if (command.equals("s")) continue;
//                showCluster(difs, tags, scanner, showBefore, showAfter, showDiff);
//            }
//        }
//    }
//
//    public static void showCluster(List<Diff> diffs, List<List<String>> tags, Scanner scanner,
//                                   boolean showBefore, boolean showAfter, boolean showDiff) {
//        for (int i = 0; i < diffs.size(); i++) {
//            Diff current = diffs.get(i);
//            System.out.println("Solution id: " + current.sessionId);
//            if (showBefore) {
//                System.out.println("--------before---------");
//                System.out.println(current.before);
//                System.out.println("----------------------");
//            }
//            if (showAfter) {
//                System.out.println("--------after---------");
//                System.out.println(current.after);
//                System.out.println("----------------------");
//            }
//            if (showDiff) {
//                if (showBefore) {
//                    System.out.println("--------diff---------");
//                    current.patches.forEach(System.out::println);
//                    System.out.println("----------------------");
//                }
//            }
//            System.out.print("Tags: ");
//            System.out.println(tags.get(i).stream().map(Object::toString).collect(Collectors.joining(", ")));
//            System.out.println("Insert s to skip cluster, n to show next solution");
//            String command = scanner.next();
//            while (!command.equals("s") && !command.equals("n")) {
//                System.out.println("Unrecognized command");
//                System.out.println("Insert s to skip cluster, n to show first solution");
//                command = scanner.next();
//            }
//            if (command.equals("s")) return;
//        }
//    }
//}
