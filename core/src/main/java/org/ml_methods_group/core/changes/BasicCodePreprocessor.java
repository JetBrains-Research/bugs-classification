package org.ml_methods_group.core.changes;

public class BasicCodePreprocessor implements CodePreprocessor {

    @Override
    public String process(String code) {
        if (code.contains("import") || code.contains("package")) {
            return code;
        } else {
            return "class MyMagicClass {\n" + code + "\n}";
        }
    }
}
