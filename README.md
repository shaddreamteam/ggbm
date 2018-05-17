# ggbm
Good game boosting machine

Планируется написать утилиту для реализации градиентного бустинга на c++. Запуск будет осуществляться в консоли с передачей параметров в command line:
* режим (обучение, применение)
* путь ко входному csv файлу для обучения или применения
* путь к выходному файлу для применения
* параметры для обучения (как минимум количество деревьев, learning rate, thread count, l2-regularization, max depth, возможно другие параметры для регуляризации, планируется поддержать две loss функции MSE и logloss для бинарной классификации)


Планируется многопоточная реализация с использованием гистограммного подхода и oblivious trees.
Минимальный план экспериментов: на датасетах Higgs и BCI с использованием функции потерь binary logloss провести обучение с использованием утилиты, а также фреймворков LightGBM, XGBoost и Catboost последних версий. Список параметров для сравнения:
* Время работы
* Пиковое потребление памяти
* Достигнутое качество на отложенной выборке

Для экспериментов будут задействованы одинаковые наборы гиперпараметров (тк мы реализуем лишь незначительный набор регуляризаций и для сокращения возможного перебора гиперпараметров фреймворков-конкурентов утилиты).
Эксперимент будет состоять в переборе гиперпараметров и последующей оценке качества с помощью кроссвалидации на 70% данных и затем обучения на этих 70% и оценке лучшей полученной в ходе кроссвалидации модели на 30% holdout выборке. Реализация кроссвалидации в утилите пока за рамками проекта.


## Использование

### Сборка
В корне директории cpp:
* mkdir build (если еще нет)
* rm -rf build/*
* cd build
* cmake ..
* make -j4
* ./cpp parameters

### Параметры
* mode = [train, predict] - режим применения, для train если задан filename_test, то после обучения будет предсказание
* filename_train - путь к файлу с данными для обучения, первой колонкой должен быть таргет
* filename_test - путь к файлу с данными для предсказания
* file_has_target = [0, 1] - есть ли в файле для предсказания колонка с таргетом
* filename_model - путь к файлу модели для сохранения в режиме обучения или для загрузки в режиме предсказания
* filename_output - путь к файлу для сохранения результатов предсказания
* threads - количество используемых потоков
* objective = [mse, logloss] - тип функции потерь
* n_estimators - количество деревьев к построению
* learning_rate - learning_rate
* depth - глубина деревьев (деревья всегда строятся данной глубины)
* lambda - L2 регуляризация на листьях
* row_subsampling - доля случайной подвыборки датасета, которая используется при построении отдельного дерева
* verbose = [0, 1] - выводить информацию о построении деревьев

Пример запуска для обучения:

./cpp mode=train threads=6 objective=logloss learning_rate=0.1 depth=6 n_estimators=400 lambda=0.0005 row_subsampling=1 filename_train=./test_model/train.csv filename_model=./test_model/model.bst

Для применения

./cpp mode=predict threads=6 filename_train=./test_model/test.csv filename_model=./test_model/model.bst filename_output=./test_model/output.csv


## Предварительные результаты:

Предварительные результаты параметров обучения на 1 потоке в сравнении с lightgbm и catboost на higgs с kaggle (250k objects) (0.7 train, 0.3 test)

| framework | Time (s) | Memory (kb) | Quality (logloss) |
| --------- | -------- | ----------- | ----------------- |
| ggbm      | 16.54    | 32976       | 0.35312           |
| lightgbm  | 10.48    | 194848      | 0.35528           |
| catboost  | 40.21    | 103448      | 0.35226           |

команды запуска:

./cpp  mode=train  threads=1 objective=logloss learning_rate=0.1 depth=6 n_estimators=400 lambda=0.0005 row_subsampling=1 filename_train=./test_model/train.csv filename_model=./test_model/model.bst

lightgbm objective=binary data=./test_model/train.csv num_leaves=64 num_threads=1 num_iterations=400

./catboost fit -f data/higgs/train.csv --loss-function Logloss --cd ../cpp/data/catboost_descriptor  --delimiter ',' --iterations 400 --thread-count 1 --depth 7 --l2-leaf-reg 0.5 --learning-rate 0.1

## Результаты:

Тесты на большом higgs в 4 потока (11М objects, 0.7 train, 0.3 test):

| framework      | depth | Time (s)    | Memory (kb) | Quality (logloss) |
| -------------- | ----- | ----------- | ----------- | ----------------- |
| ggbm           |   3   | 190.02      | **1103776** | 0.53555           |
| lightgbm       |   3   | **84.91**   | 4561384     | **0.52068**       |
| xgboost(hist)  |   3   | 177.30      | 3946152     | 0.52768           |
| catboost       |   3   | 372.50      | 2740428     | 0.52303           |
| ggbm           |   6   | 356.76      | **1108420** | 0.52116           |
| lightgbm       |   6   | **227.54**  | 4560888     | 0.50695           |
| xgboost(hist)  |   6   | 320.28      | 3963880     | 0.51559           |
| catboost       |   6   | 555.58      | 2744352     | **0.50108**       |
| ggbm           |   9   | 756.98      | **1102476** | 0.51810           |
| lightgbm       |   9   | 632.53      | 4557716     | 0.49210           |
| xgboost(hist)  |   9   | **438.34**  | 4046596     | **0.482425**      |
| catboost       |   9   | 787.98      | 2769672     | 0.48806           |

Наша реализация градиентного бустинга работает сравнимо с рассмотренными библитоеками по качеству и скорости. Догнать lightgbm не удалось, для этого, видимо, надо улучшать версию data parallel или существенно перерабатывать feature parallel. У нас сильно медленнее загружаются данные: для большого датасета 40 секунд против 8 у lightgbm, 4 у XGBoost histogram и 20 у Catboost (скорее всего, первые два параллельно парсят строки, чего мы не делаем), но оптимизацией загрузки с диска мы оставляем за рамками проекта (тк заметили это в самом конце). В итоге, если вычесть время загрузки из таблицы с результатами, наша реализация работает везде быстрее, чем Catboost, не медленней XGBoost histogram на малой и средней глубине, а так же приближается к LightGBM на большой.

Отедльно стоило бы отметить неожиданную скорость XGBoost на большой глубине и неожиданно низкую скорость Catboost. Catboost по умолчанию для бинарной классификации испльзует 10 шагов метода Ньютона для бинарной классификации, однако, если делать один градиентный шаг, это не сильно уменьшает время в сравнении с другими бустингами, поэтому мы оставили параметр по умолчанию.

### Измерения:
* Время и память измерялись утилитой time с параметрами -f "%e %M"

## Отчет

### Алгоритм

* Загружаются данные
* В разных потоках фичи сортитуются, считаются количества уникальных элементов и идет разбиения по бинам так, чтобы в каждом бине было по возможности поровну элементов
* Делается базовое предсказание средним и начинается построение деревьев
* Для каждого элемента считается градиент и гессиан
* Гистограммы gain строятся для разных фич разными потоками
* Строятся кумулятивные суммы gain для гистограмм (при построении учитывается, что кумулятивные суммы gain для листа с большим количество элементов можно получить отняв кумулятивные суммы листа с меньшим количеством элементов из кумулятивных сумм предка) для каждой фичи выбирается сплит с наилучшим gain
* После определения сплита параллельно создаются новые листья, параллельность заключается в отдельной обработке индексов элементов содержащихся в листе, причем равномерно бъется весь датасет, так что части одного листа предка могут обрабатываться разными потоками.
* В случае если лист пустой - ему присваивается вес предка, это упрощает алгоритм и позволяет верно классифицировать тестовый элемент, который попадает по сплитам в лист, в котором не было train элементов
* После построения дерева учитывается, что в листах на последнем уровне уже содержится по сути предсказание для этого дерева и поэтому можно быстро обновить текущее предсказание для train dataset
* После построения дерева параллельно обновляются градиенты и гессианы для объектов в train dataset 

### Цель

Основной целью было получить хорошую скорость работы при сравнимом качестве (поскольку у нас параметров меньше, а для серьезных бустингов куча параметров задана по умолчанию) и адекватно сравнить с другими библиотеками. Один из способов это сделать - выставить порог достижения некоторого качества, и посмотреть насколько быстро мы его достигаем, однако это очень сильно завязано на тюнинге бустингов, и конкурировать в возможности затюнить бустинг с библиотеками-конкурентами - не лучшая идея, поэтому мы пошли по другому пути и решили зафиксировать структуру дерева и подобрать наилучшие параметры (из доступных нам и не влияющих на скорость построения).

### Success story

Сначала мы сделали однопоточный прототип на Python как baseline. При разработке рабочего решения выбрали схему паралеллизации feature parallel. Была очень большая проблема с производительностью в начале, решали ее несколько дней, оказалось, что мы считали гистограмму (самая нагруженная функция) во внутреннем цикле вместо внешнего и простой перенос нескольких строк наверх ускорил построение в 200 раз. Затем мы много занимались оптимизациями в однопоточном режиме и когда подошел срок, оказалось что многопоточная версия очень плохо ускоряет обучение. Дело в том, что мы в основном проверяли многопоточную версию в тот момент, когда у нас была ошибка с производительностью и тогда потоки ускоряли ровнов n раз, но после того как мы исправили ошибку, оказалось, что у реализации появились bottleneck в других местах, чего мы не узнали. Уже после дедлайна занялись разработкой data parallel (ветка data_parallel_trial), переписали кучу кода, но в итоге оказалось что двухпоточная версия в 8 раз медленнее однопоточной, а 4 потока в 4 раз медленнее одного :)), решили развивать дальше feature parallel - добавив параллельность везде, где можно - в итоге удалось ускорить обучение для 2-х потоков в 1.7 раза, а для 4 потоков в 3.3 раза, что вроде приемлемо. 

### PS

BCI датасет оказался каким-то странным, на kaggle существующие решения обучаются на 5440 примерах, поэтому решили на нем не тестить, а сделать все на маленьком и большом Higgs.

