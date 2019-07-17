# bugs-classification

## Сборка

git clone https://github.com/ml-in-programming/bugs-classification.git

cd bugs-classification

gradlew build

Запускаемый jar-файл: build\libs\bugs-classification-v1.jar

war-файл для сервера: server\build\libs\server-v1.war

## Консольный интерфейс

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

4. prepare _src_ _solutions_ _dst_
Преобразует размеченные данные в формат, нужный для быстрого создания классификатора.

| Аргумент  | Значение |
| :------------- | :------------- |
| _src_ | Файл с полученными кластерами, полученный с помощью команды _cluster_.  |
| _solutions_ | Файл с предобработанными данными, полученный с помощью команды parse |
| _dst_ | Итоговый файл с размеченными данными. |

Пример: `java -jar build\libs\bugs-classification-v1.jar prepare marks.tmp solutions.tmp prepared.tmp`


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

## Запуск сервера

В файле _bugs-classification\server\src\main\resources\data.txt_ прописан путь, откуда сервер попытается загрузить данные.
Каждая папка, название которой состоит только из цифр, будет интерпретирована как подготовленные данные для какой-то задачи.
Если в этом файле прописан путь path и есть подготовленные данные для задач с номерами (step\_id) _239566_ и _566239_, 
то ожидается следующее расположение файлов:
```file
path
    239566
        solutions.tmp
        prepared.tmp
    566239
        solutions.tmp
        prepared.tmp
```
После сбоки проекта, war файл будет лежать в server\build\libs.
Сервер поддерживает два вида запросов.
### Запрос подсказки
Позволяет получить подсказку для неправильного решения.

**Формат запроса следующий:**
```url
ip:port/server-v1/webapi/bugs-classification/hint?problem=*step_id*&code=*solution_code*
```
| Аргумент  | Значение |
| :------------- | :------------- |
| _problem_ | Идентификатор задачи (step\_id)  |
| _code_ | Текст решения |

**Формат ответа следующий:**
```json
{"hint":*hint_text*,"confidence":*confidence*,"errorMessage":*error_message*,"status":*status*,"time":*time*}
```
| Аргумент  | Значение |
| :------------- | :------------- |
| _hint_ | Текст подсказки. Если что-то пошло не так, то поле будет пустым.  |
| _confidence_ | Уверенность классификатора в ответе |
| _errorMessage_ | Сообщение об ошибке. Если запрос был успешно обработан, то поле будет пустым. |
| _status_ | Статус обработки запроса. Может принимать два значения: OK и ERROR |
| _time_ | Время обработки запроса на сервере |

Пример: для данного решения задачи с номером 239566
```java
void foo() {
}
```
запрос будет выглядеть так

```
http://localhost:8080/server-v1/webapi/bugs-classification/hint?problem=239566&code=void+foo%28%29+%7B%0D%0A%7D
```

а ответ так

```json
{"hint":"Похоже удача сегодня не на вашей стороне","confidence":1.0,"errorMessage":"","status":"OK","time":556}
```

### Запрос доступных классификаторов

Позволяет узнать список задач, для которых доступна функция получения подсказок.

**Формат запроса следующий:**
```url
ip:port/server-v1/webapi/bugs-classification/classifiers
```

**Формат ответа следующий:**
```json
[*problem_1*, *problem_2*, ...]
```

Например
```json
[47538,53619,55715,47334,58088,53676]
```

