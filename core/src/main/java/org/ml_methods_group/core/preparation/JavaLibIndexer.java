package org.ml_methods_group.core.preparation;

import org.ml_methods_group.core.Index;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLConnection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.ml_methods_group.core.preparation.LabelType.KEY_WORD;
import static org.ml_methods_group.core.preparation.LabelType.PRIMITIVE_TYPE;
import static org.ml_methods_group.core.preparation.LabelType.MODIFIER;

public class JavaLibIndexer {
    private final URL url;
    private final Pattern parsingRegexp;
    private final int classNameGroupIndex;
    private final int packageNameGroupIndex;
    private final Pattern charsetRegexp = Pattern.compile("text/html;\\s+charset=([^\\s]+)\\s*");

    public JavaLibIndexer(URL url, String regexp, int classNameGroupIndex, int packageNameGroupIndex) {
        this.url = url;
        this.parsingRegexp = Pattern.compile(regexp);
        this.classNameGroupIndex = classNameGroupIndex;
        this.packageNameGroupIndex = packageNameGroupIndex;
    }

    public JavaLibIndexer() throws MalformedURLException {
        this(new URL("https://docs.oracle.com/javase/8/docs/api/allclasses-noframe.html"),
                "<li><a href=\"[^\"]+\"\\s+title=\"class\\s+in\\s+([.a-zA-Z0-9]+)\">([.a-zA-Z0-9]+)</a></li>",
                2, 1);
    }

    public void index(Index<String, LabelType> index) throws IOException, ClassNotFoundException {
        index.clean();
        final Set<String> inserted = new HashSet<>();
        indexKeyWords(index, inserted);
        indexClasses(index, inserted);
    }

    private void tryInsert(String key, LabelType value, Index<String, LabelType> index, Set<String> inserted) {
        if (!inserted.contains(key)) {
            inserted.add(key);
            index.insert(key, value);
        }
    }

    private void indexKeyWords(Index<String, LabelType> index, Set<String> inserted) throws IOException {
        tryInsert("void", PRIMITIVE_TYPE, index, inserted);
        tryInsert("boolean", PRIMITIVE_TYPE, index, inserted);
        tryInsert("byte", PRIMITIVE_TYPE, index, inserted);
        tryInsert("char", PRIMITIVE_TYPE, index, inserted);
        tryInsert("short", PRIMITIVE_TYPE, index, inserted);
        tryInsert("int", PRIMITIVE_TYPE, index, inserted);
        tryInsert("float", PRIMITIVE_TYPE, index, inserted);
        tryInsert("long", PRIMITIVE_TYPE, index, inserted);
        tryInsert("double", PRIMITIVE_TYPE, index, inserted);

        tryInsert("public", MODIFIER, index, inserted);
        tryInsert("protected", MODIFIER, index, inserted);
        tryInsert("private", MODIFIER, index, inserted);
        tryInsert("static", MODIFIER, index, inserted);
        tryInsert("default", MODIFIER, index, inserted);
        tryInsert("final", MODIFIER, index, inserted);
        tryInsert("abstract", MODIFIER, index, inserted);
        tryInsert("synchronized", MODIFIER, index, inserted);
        tryInsert("native", MODIFIER, index, inserted);
        tryInsert("volatile", MODIFIER, index, inserted);
        tryInsert("transient", MODIFIER, index, inserted);
        tryInsert("strictfp", MODIFIER, index, inserted);

        tryInsert("for", KEY_WORD, index, inserted);
        tryInsert("do", KEY_WORD, index, inserted);
        tryInsert("while", KEY_WORD, index, inserted);
        tryInsert("if", KEY_WORD, index, inserted);
        tryInsert("else", KEY_WORD, index, inserted);
        tryInsert("switch", KEY_WORD, index, inserted);
        tryInsert("case", KEY_WORD, index, inserted);
        tryInsert("break", KEY_WORD, index, inserted);
        tryInsert("continue", KEY_WORD, index, inserted);
        tryInsert("return", KEY_WORD, index, inserted);

        tryInsert("class", KEY_WORD, index, inserted);
        tryInsert("interface", KEY_WORD, index, inserted);
        tryInsert("enum", KEY_WORD, index, inserted);
        tryInsert("implements", KEY_WORD, index, inserted);
        tryInsert("extends", KEY_WORD, index, inserted);
        tryInsert("instanceof", KEY_WORD, index, inserted);
        tryInsert("this", KEY_WORD, index, inserted);
        tryInsert("super", KEY_WORD, index, inserted);

        tryInsert("throw", KEY_WORD, index, inserted);
        tryInsert("throws", KEY_WORD, index, inserted);
        tryInsert("try", KEY_WORD, index, inserted);
        tryInsert("catch", KEY_WORD, index, inserted);
        tryInsert("finally", KEY_WORD, index, inserted);

        tryInsert("import", KEY_WORD, index, inserted);
        tryInsert("package", KEY_WORD, index, inserted);

        tryInsert("assert", KEY_WORD, index, inserted);
        tryInsert("new", KEY_WORD, index, inserted);

        tryInsert("goto", KEY_WORD, index, inserted);
        tryInsert("const", KEY_WORD, index, inserted);
    }

    private void indexClasses(Index<String, LabelType> index, Set<String> inserted) throws IOException {
        final Map<String, String> classes = parseClasses();
        for (Map.Entry<String, String> entry : classes.entrySet()) {
            tryInsert(entry.getKey(), LabelType.STANDARD_CLASS_FULL_NAME, index, inserted);
            tryInsert(entry.getValue(), LabelType.STANDARD_CLASS_NAME, index, inserted);
            indexMethods(entry.getKey(), index, inserted);
        }
    }

    private void indexMethods(String className,
                              Index<String, LabelType> index, Set<String> inserted) {
        final Class<?> aClass;
        try {
            aClass = Class.forName(className);
            for (Method method : aClass.getDeclaredMethods()) {
                if ((method.getModifiers() & Modifier.PUBLIC) != 0) {
                    tryInsert(method.getName(), LabelType.STANDARD_METHOD_NAME, index, inserted);
                }
            }
        } catch (ClassNotFoundException ignored) {
        }
    }

    private Map<String, String> parseClasses() throws IOException {
        final String text = load();
        final Matcher matcher = parsingRegexp.matcher(text);
        final Map<String, String> classes = new HashMap<>();
        while (matcher.find()) {
            final String packageName = matcher.group(packageNameGroupIndex);
            final String className = matcher.group(classNameGroupIndex);
            classes.put(packageName + "." + className, className);
        }
        return classes;
    }

    private String load() throws IOException {
        final URLConnection con = url.openConnection();
        final Matcher m = charsetRegexp.matcher(con.getContentType());
        final String charset = m.matches() ? m.group(1) : "ISO-8859-1";
        StringBuilder buf = new StringBuilder();
        try (Reader r = new InputStreamReader(con.getInputStream(), charset)) {
            while (true) {
                int ch = r.read();
                if (ch < 0)
                    break;
                buf.append((char) ch);
            }
        }
        return buf.toString();
    }
}
