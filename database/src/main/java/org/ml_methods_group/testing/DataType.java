package org.ml_methods_group.testing;

enum DataType {
    INTEGER("INTEGER", false, int.class, Integer.class),
    DOUBLE("FLOAT", false, double.class, Double.class),
    LONG("BIGINT", false, long.class, Long.class),
    BOOLEAN("BOOL", false, boolean.class, Boolean.class),
    STRING("BYTEA", false, String.class),
    ENUM("TEXT", false),
    SERIAL("SERIAL", true);

    private final Class[] supported;
    public final String sqlName;
    public final boolean service;

    DataType(String sqlName, boolean service, Class... supported) {
        this.sqlName = sqlName;
        this.service = service;
        this.supported = supported;
    }

    public static DataType getDefaultTypeFor(Class<?> template) {
        for (DataType dataType : values()) {
            for (Class supported : dataType.supported) {
                if (supported.equals(template)) {
                    return dataType;
                }
            }
        }
        if (Enum.class.isAssignableFrom(template)) {
            return ENUM;
        }
        return null;
    }
}
