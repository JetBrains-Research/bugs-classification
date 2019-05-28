# bugs-classification

## Сборка

git clone https://github.com/ml-in-programming/bugs-classification.git

cd bugs-classification

gradle build

Получившийся jar-ник: build\libs\bugs-classification-v1.jar

## Запуск

java -jar bugs-classification-v1.jar [команда]


1. parse [csv-файл] [куда сохранить датасет]   --  преобразует csv файл в используемый в проекте формат

Пример: `java -jar bugs-classification-v1.jar parse data.csv data.tmp`

2. cluster [файл с данными] [куда сохранить кластеры]   --   кластеризует неправильные решения  

Пример: `java -jar bugs-classification-v1.jar cluster data.tmp clusters.tmp

3. mark [файд с кластерами] [файл, куда записать результат] [сколько примеров показывать из кластера] [сколько кластеров показать]    --   позволяет разметить кластеры

Пример: `java -jar bugs-classification-v1.jar show clusters.tmp marks.tmp 5 40`

4. classify [файл с датасетом] [файл с размеченными кластерами] [файл решением]    --   классифицирует новое решение



Формат исходного csv файла ожидается такой:
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
