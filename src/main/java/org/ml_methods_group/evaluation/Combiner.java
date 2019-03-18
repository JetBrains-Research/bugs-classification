package org.ml_methods_group.evaluation;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.*;

public class Combiner {

    public static class Test {
        final double precision;
        final double recall;
        final double threshold;

        public Test(double precision, double recall, double threshold) {
            this.precision = precision;
            this.recall = recall;
            this.threshold = threshold;
        }

        public double getPrecision() {
            return precision;
        }

        public double getRecall() {
            return recall;
        }

        public double getThreshold() {
            return threshold;
        }
    }

    public static void main(String[] args) throws FileNotFoundException {
        File dir = new File("results");
        final HashMap<String, List<Test>> results = new HashMap<>();
        for (File f : dir.listFiles()) {
            Scanner scanner = new Scanner(f);
            while (scanner.hasNextLine()) {
                final String line = scanner.nextLine();
                final String[] tokens = line.replaceAll("'", "").split(",");
                final String prefix = tokens[0] + "," + tokens[1] + "," + tokens[2] + "," + tokens[3] + "," +
                        f.getName().replaceAll("\\..*", "") + "," + tokens[7];
                System.out.println(prefix);
                final List<Test> container = results.computeIfAbsent(prefix, x -> new ArrayList<>());
                container.add(new Test(parseDouble(tokens[6]), parseDouble(tokens[5]), parseDouble(tokens[4])));
            }
        }
        for (String s : results.keySet()) {
            if (s.startsWith("def_vec,0.4,20,k-nearest-5,BOW20000")) {
                System.out.println(s);
            }
        }
        String prefix = "def_vec,0.4,20,k-nearest-5,BOW20000,0.675";
        final List<Test> points = results.get(prefix);

        points.sort(Comparator.comparingDouble(Test::getRecall)
                .thenComparingDouble(x -> -x.precision)
                .thenComparingDouble(x -> -x.threshold));

        for (Test t : points) {
            System.out.println("(" + t.recall + ", " + t.precision + ")");
        }

//        try (PrintWriter out = new PrintWriter("aucs.csv")) {
//            for (Map.Entry<String, List<Test>> entry : results.entrySet()) {
//                final List<Test> points = entry.getValue();
//                out.println(entry.getKey() + "," + getAUC(points));
//            }
//        }
    }

    public static double getAUC(List<Test> points) {
        points.sort(Comparator.comparingDouble(Test::getRecall)
                .thenComparingDouble(Test::getPrecision)
                .thenComparingDouble(Test::getThreshold));
        double auc = 0;
        double prevPrecision = 1;
        double prevRecall = 0;
        for (Test point : points) {
            auc += (point.recall - prevRecall) * (point.precision + prevPrecision) / 2;
            prevPrecision = point.precision;
            prevRecall = point.recall;
        }
        auc += (1 - prevRecall) * prevPrecision / 2;
        return auc;
    }

    public static double parseDouble(String s) {
        return Double.parseDouble(s);
    }
}
