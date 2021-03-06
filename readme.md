## Задача

Дан граф G(V,E) с компонентой связности равной 1. Граф задается матрицей расстояний между вершинами distance_matrix, где distance_matrix[i][j] - расстояние между i и j вершинами.

Если distance_matrix[i][j] = 0, то не существует пути из i вершины в j.

Необходимо пройти по всем вершинам и вернутся в исходную так, чтобы суммарный путь был минимальным. Начало пути следует начинать с первой вершины. Разрешается проходить по одной вершине несколько раз.

## Идея решения

Максимально оптимизируем полный перебор.

1. С помощью алгоритма Дейкстры находим минимально возможное расстояние от всех вершин до первой. Запоминаем это расстояние и последовательность вершин.

2. Сразу задаем верхнюю оценку суммарного пути (Как удвоенные суммы путей, ранее найденных алгоритмом Дейкстры)

3. Заранее сортируем рёбра по их стоимости и порядку

4. При обходе приоритет отдаем ранее не посещенным вершинам

_Даже при этих условиях полный перебор не уложится в лимит времени, если в графе из 10 вершин слишком много ребер. В тестах таких графов нет._