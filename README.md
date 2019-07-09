# bugs-classification

## Сборка

git clone https://github.com/ml-in-programming/bugs-classification.git

cd bugs-classification

gradlew build

Запускаемый jar-файл: build\libs\bugs-classification-v1.jar

war-файл для сервера: server\build\libs\server-v1.war

## Запуск

java -jar build\libs\bugs-classification-v1.jar [команда]


1. parse _src_ _dst_  
Преобразует csv файл в используемый в проекте формат

| Аргумент  | Значение |
| :------------- | :------------- |
| _src_ | Исходный csv-файл с данными. Ожидаемый формат описан ниже.  |
| _dst_ | Итоговый файл с предобработанными данными для дальнейшей работы. |
| _step\_id_ | ID задачи для фильтрации решений. |

Пример: `java -jar build\libs\bugs-classification-v1.jar parse data.csv solutions.tmp 239566`

2. cluster _src_ _dst_  
Кластеризует неправильные решения  

| Аргумент  | Значение |
| :------------- | :------------- |
| _src_ | Файл с предобработанными данными, полученный с помощью команды _parse_.  |
| _dst_ | Итоговый файл с кластеризованными исправлениями. |

Пример: `java -jar build\libs\bugs-classification-v1.jar cluster solutions.tmp clusters.tmp

3. mark _src_ _dst_ _n\_show_ _n\_clusters_  
Позволяет разметить кластеры. Размечается n самых больших кластеров.

| Аргумент  | Значение |
| :------------- | :------------- |
| _src_ | Файл с полученными кластерами, полученный с помощью команды _cluster_.  |
| _dst_ | Итоговый файл с размеченными данными. |
| _n\_show_ | Количество элементов кластера, которые будут показываться для примера |
| _n\_clusters_ | Количество кластеров, которые будут размечены. |

Пример: `java -jar build\libs\bugs-classification-v1.jar mark clusters.tmp marks.tmp 5 40`

## Формат исходного csv файла 
Ожидается такой формат (порядок столбцов значения не имеет):
```csv
step_id,user_id,submission_code,is_passed,timestamp
239,566,"public class HelloWorld {
	public static void main(String[] args) {
		System.out.println(""Hello, World!"");
	}
}",False,1442181616
239,566,"class HelloWorld {
  public static void main(String[] args) {
    System.out.println(""Hello, World!"");
  }
}",False,1442182020
```
