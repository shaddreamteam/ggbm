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


## Результаты:

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

Тесты на большом higgs в 4 потока (11М objects, 0.7 train, 0.3 test):

| framework      | depth | Time (s)    | Memory (kb) | Quality (logloss) |
| -------------- | ----- | ----------- | ----------- | ----------------- |
| ggbm           |   3   | 190.02      | **1103776** | 0.53555           |
| lightgbm       |   3   | **84.91**   | 4561384     | **0.52068**       |
| xgboost(hist)  |   3   | 177.30      | 3946152     | 0.52768           |
| ggbm           |   6   | 356.76      | **1108420** | 0.52116           |
| lightgbm       |   6   | **227.54**  | 4560888     | **0.50695**       |
| xgboost(hist)  |   6   | 320.28      | 3963880     | 0.51559           |
| ggbm           |   9   | 756.98      | **1102476** | 0.51810           |
| lightgbm       |   9   | 632.53      | 4557716     | 0.49210           |
| xgboost(hist)  |   9   | **438.34**  | 4046596     | **0.482425**      |


Измерения:
* Время: команда time
* Память: команда time -f "%M"

