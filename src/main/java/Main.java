import org.ml_methods_group.clusterization.HAC;
import org.ml_methods_group.core.SolutionDiff;
import org.ml_methods_group.core.Utils;
import org.ml_methods_group.core.preparation.ChangesBuilder;
import org.ml_methods_group.core.preparation.LabelType;
import org.ml_methods_group.core.selection.CenterSelector;
import org.ml_methods_group.core.testing.ExternalTester;
import org.ml_methods_group.core.testing.InternalTester;
import org.ml_methods_group.core.testing.Pair;
import org.ml_methods_group.core.testing.TestingUtils;
import org.ml_methods_group.core.vectorization.*;
import org.ml_methods_group.database.ChangeCodeIndex;
import org.ml_methods_group.database.LabelIndex;
import org.ml_methods_group.database.StandardLabelIndex;
import org.ml_methods_group.database.TestPairIndex;
import org.ml_methods_group.database.proxy.ProxySolutionDatabase;
import org.ml_methods_group.ui.ConsoleIO;
import org.ml_methods_group.ui.UtilsUI;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.sql.SQLException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.ml_methods_group.core.Utils.defaultStrategies;
import static org.ml_methods_group.core.selection.CenterSelector.Mode.MAX;

public class Main {

    private static final int PROBLEM = 58088;

    public static void main(String[] args) throws SQLException, IOException, ClassNotFoundException {
//        markTests();
        StandardLabelIndex index = new StandardLabelIndex();
//        JavaLibIndexer indexer = new JavaLibIndexer();
//        indexer.index(index);
//        PrintWriter out = new PrintWriter("out.csv");
//        ProxySolutionDatabase database = new ProxySolutionDatabase();
//        LabelIndex labels = new LabelIndex();
//        ChangeCodeIndex codes = new ChangeCodeIndex();
////        Utils.loadDatabase(Main.class.getResourceAsStream("/dataset.csv"), database,
////                index.getIndex(), PROBLEM, labels);
//        final Map<String, Integer> dictionary = labels.getIndex()
//                .keySet()
//                .stream()
//                .filter(wrapper -> wrapper.getType() != LabelType.UNKNOWN)
//                .collect(Collectors.toMap(LabelWrapper::getLabel, LabelWrapper::getId));
//        VectorTemplate template = Utils.generateTemplate(database, codes, defaultStrategies(dictionary),
//                VectorTemplate.BasePostprocessors.NONE, 2, Integer.MAX_VALUE);
//        final List<SolutionDiff> solutions = database.getDiffs();
//        System.out.println(solutions.size());
//        Collections.shuffle(solutions, new Random(239566));
//        final List<SolutionDiff> train = solutions.subList(0, 1000);
//        final List<SolutionDiff> test = solutions.subList(1000, solutions.size());
//        final NearestSolutionVectorizer vectorizer = new NearestSolutionVectorizer(train, template,
//                new ChangesBuilder());

//        final List<Wrapper> wrappers = train.stream()
////                .map(diff -> new Wrapper(template.process(diff.getChanges()), diff.getSessionId()))
//                .map(diff -> {
//                    try {
//                        System.out.println("Next");
//                        return new Wrapper(vectorizer.process(diff.getCodeBefore()), diff.getSessionId());
//                    } catch (Exception e) {
//                        return null;
//                    }
//                })
//                .filter(wrapper -> MathUtils.norm(wrapper.vector) > 0.00001)
//                .collect(Collectors.toList());
//        PrintWriter vectors = new PrintWriter("vectors.txt");
//        for (Wrapper wrapper : wrappers) {
//            vectors.println(wrapper.sessionId + " " + Arrays.toString(wrapper.vector));
//        }
//        vectors.close();
        List<Wrapper> wrappers = parse("vectors.txt");
        System.out.println(wrappers.size());
        MathUtils.standardize(wrappers.stream().

                map(w -> w.vector).

                collect(Collectors.toList()));
        final List<List<Wrapper>> lists;
        try (
                HAC<Wrapper> clusterer = new HAC<>(0.2, 30, Wrapper::euclideanDistance))

        {
            lists = clusterer.buildClusters(wrappers);
        }
        System.out.println(lists.size());
        lists.stream()
                .map(List::size)
                .forEachOrdered(System.out::println);
        InternalTester tester1 = new InternalTester(Wrapper::squaredEuclideanDistance);
        ExternalTester tester2 = new ExternalTester(new TestPairIndex());
        System.out.println(tester1.test(lists));
        System.out.println(tester2.test(lists));
//        final Map<Wrapper, Integer> mapping = new HashMap<>();
//        PrintWriter cache = new PrintWriter("cache.txt");
//        for (List<Wrapper> list : lists) {
//            for (Wrapper wrapper : list) {
//                cache.print(wrapper.sessionId + " ");
//                cache.println();
//            }
//        }
//        cache.close();
        //        final Classifier<Wrapper> classifier = new KNearestNeighbors(5);
//        classifier.train(mapping);
        ConsoleIO console = new ConsoleIO();
        final CenterSelector<Wrapper> selector = new CenterSelector<>(Wrapper::squaredEuclideanDistance, 0.2, MAX);
//        for (List<Wrapper> cluster : lists) {
//            console.write("Next cluster: size " + cluster.size());
//            if (cluster.size() < 3) continue;
//            for (Wrapper sample : selector.findRepresenter(15, cluster)) {
//                console.write("Show next sample?");
//                if (console.expect("+", "-") != 0) {
//                    break;
//                }
//                console.write(database.getDiff(sample.sessionId));
//            }
//        }


//        final List<String> marks = UtilsUI.markClusters(lists, selector, database, console, l -> l.size() > 5);
//        final HintGenerator generator = new HintGenerator(vectorizer, classifier);
//        for (int i = 0; i < marks.size(); i++) {
//            generator.setHint(i, marks.get(i));
//        }
//        console.write("Start testing:");
//        for (int i = 0; i < 20; i++) {
//            final SolutionDiff example = test.get(i);
//            console.write("------------------------next-case-----------------------------");
//            console.write(example);
//            console.write(generator.getTip(example.getCodeBefore()));
//            console.readLine();
//        }
    }

    private static List<Wrapper> parse(String file) throws IOException {
        return Files.lines(Paths.get(file))
                .map(line -> {
                    line = line.replaceAll("[,\\[\\]]", " ");
                    final String[] tokens = line.split("\\s+");
                    final int id = Integer.parseInt(tokens[0]);
                    final double[] array = Arrays.stream(tokens, 1, tokens.length)
                            .mapToDouble(Double::parseDouble)
                            .toArray();
                    return new Wrapper(array, id);
                })
                .collect(Collectors.toList());

    }

    public static void writeCluster(List<Wrapper> wrappers, VectorTemplate template) throws FileNotFoundException {
        PrintWriter writer = new PrintWriter("vectors.csv");
        for (int i = 0; i < wrappers.get(0).vector.length; i++) {
            final int index = i;
            final String line = wrappers.stream()
                    .mapToDouble(wrapper -> wrapper.vector[index])
                    .mapToObj(Double::toString)
                    .collect(Collectors.joining(","));
            writer.println(line);
        }
        writer.close();
    }

    public static void markTests() throws IOException, ClassNotFoundException {
        final Random random = new Random(239566);

        StandardLabelIndex index = new StandardLabelIndex();
//        JavaLibIndexer indexer = new JavaLibIndexer();
//        indexer.index(index);
        ProxySolutionDatabase database = new ProxySolutionDatabase();
        LabelIndex labels = new LabelIndex();
        ChangeCodeIndex codes = new ChangeCodeIndex();
//        Utils.loadDatabase(Main.class.getResourceAsStream("/dataset.csv"), database,
//                index.getIndex(), PROBLEM, labels);
        final Map<String, Integer> dictionary = labels.getIndex()
                .keySet()
                .stream()
                .filter(wrapper -> wrapper.getType() != LabelType.UNKNOWN)
                .collect(Collectors.toMap(LabelWrapper::getLabel, LabelWrapper::getId));
        VectorTemplate template = Utils.generateTemplate(database, codes, defaultStrategies(dictionary),
                VectorTemplate.BasePostprocessors.RELATIVE, 2, Integer.MAX_VALUE);
        final List<Wrapper> wrappers = database.getDiffs().stream()
                .map(d -> new Wrapper(template.process(d.getChanges()), d.getSessionId()))
                .collect(Collectors.toList());
        Collections.shuffle(wrappers, random);
        final List<Pair<SolutionDiff>> cases = TestingUtils.generatePairs(wrappers.subList(0, 500),
                Wrapper::euclideanDistance, 100, random)
                .stream()
                .map(p -> new Pair<>(database.getDiff(p.first.sessionId), database.getDiff(p.second.sessionId)))
                .collect(Collectors.toList());
        UtilsUI.markPairs(cases, new ConsoleIO(), new TestPairIndex());
    }
}
