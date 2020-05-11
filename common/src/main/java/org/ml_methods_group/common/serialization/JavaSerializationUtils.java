package org.ml_methods_group.common.serialization;

import java.io.*;
import java.nio.file.Path;

public class JavaSerializationUtils {
    public static void storeObject(Object object, Path path) throws IOException {
        final Path directory = path.getParent();
        if (directory != null && !directory.toFile().exists() && !directory.toFile().mkdirs()) {
            throw new IOException("Failed to create parent directories: " + directory.toString());
        }
        try (FileOutputStream fos = new FileOutputStream(path.toFile());
             ObjectOutputStream oos = new ObjectOutputStream(fos)) {
            oos.writeObject(object);
        }
    }

    public static <T> T loadObject(Class<T> aClass, Path path) throws IOException {
        try (FileInputStream fis = new FileInputStream(path.toFile());
             ObjectInputStream ois = new ObjectInputStream(fis)) {
            return aClass.cast(ois.readObject());
        } catch (ClassNotFoundException e) {
            throw new IOException("Unexpected object type!", e);
        }
    }
}
