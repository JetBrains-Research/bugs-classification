import org.ml_methods_group.clusterization.HAC;
import org.ml_methods_group.core.SolutionDiff;
import org.ml_methods_group.core.Utils;
import org.ml_methods_group.core.preparation.LabelType;
import org.ml_methods_group.core.testing.ExternalTester;
import org.ml_methods_group.core.testing.InternalTester;
import org.ml_methods_group.core.testing.Pair;
import org.ml_methods_group.core.testing.TestingUtils;
import org.ml_methods_group.core.vectorization.LabelWrapper;
import org.ml_methods_group.core.vectorization.MathUtils;
import org.ml_methods_group.core.vectorization.VectorTemplate;
import org.ml_methods_group.core.vectorization.Wrapper;
import org.ml_methods_group.database.ChangeCodeIndex;
import org.ml_methods_group.database.LabelIndex;
import org.ml_methods_group.database.StandardLabelIndex;
import org.ml_methods_group.database.TestPairIndex;
import org.ml_methods_group.database.proxy.ProxySolutionDatabase;
import org.ml_methods_group.ui.ConsoleIO;
import org.ml_methods_group.ui.UtilsUI;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.sql.SQLException;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;

import static org.ml_methods_group.core.Utils.defaultStrategies;

public class Main {

    private static final int PROBLEM = 58088;

    public static void main(String[] args) throws SQLException, IOException, ClassNotFoundException {
//        markTests();
        StandardLabelIndex index = new StandardLabelIndex();
//        JavaLibIndexer indexer = new JavaLibIndexer();
//        indexer.index(index);
        PrintWriter out = new PrintWriter("out.csv");
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
                VectorTemplate.BasePostprocessors.STANDARD, 2, Integer.MAX_VALUE);
        final List<SolutionDiff> solutions = database.getDiffs();
        System.out.println(solutions.size());
        Collections.shuffle(solutions, new Random(239566));
        final List<SolutionDiff> train = solutions.subList(0, 1000);
        final List<SolutionDiff> test = solutions.subList(1000, solutions.size());
        final List<Wrapper> wrappers = train.stream()
                .map(diff -> new Wrapper(template.process(diff.getChanges()), diff.getSessionId()))
                .filter(wrapper -> MathUtils.norm(wrapper.vector) > 0.00001)
                .collect(Collectors.toList());
        final List<List<Wrapper>> lists;
        try (HAC<Wrapper> clusterer = new HAC<>(0.45, 30, Wrapper::squaredEuclideanDistance)) {
            lists = clusterer.buildClusters(wrappers);
        }
        InternalTester tester1 = new InternalTester(Wrapper::squaredEuclideanDistance);
        ExternalTester tester2 = new ExternalTester(new TestPairIndex());
        System.out.println(tester1.test(lists));
        System.out.println(tester2.test(lists));
//        final Map<Wrapper, Integer> mapping = new HashMap<>();
//        for (List<Wrapper> list : lists) {
//            out.write(list.size() + "\n");
//            if (list.size() > 10 && list.size() < 20) {
//                writeCluster(list, template);
//            }
//        }
//        out.close();
//        final Classifier<Wrapper> classifier = new KNearestNeighbors(5);
//        classifier.train(mapping);
//        final NearestSolutionVectorizer vectorizer = new NearestSolutionVectorizer(train, template,
//                new ChangesBuilder());
//        ConsoleIO console = new ConsoleIO();
//        final CenterSelector<Wrapper> selector = new CenterSelector<>(Wrapper::squaredEuclideanDistance, 0.2, MAX);
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
