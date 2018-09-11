# Дан граф G(V,E) с компонентой связности равной 1. Граф задается
# матрицей расстояний между вершинами distance_matrix, где
# distance_matrix[i][j] - расстояние между i и j вершинами.
# Если distance_matrix[i][j] = 0, то не существует пути из i вершины в j.
# Необходимо пройти по всем вершинам и вернутся в исходную так, чтобы
# суммарный путь был минимальным. Начало пути следует начинать с первой
# вершины. Разрешается проходить по одной вершине несколько раз.
#
# На вход подается в первой строке количество вершин n≤10,
# далее следует матрица смежности, на выходе необходимо вернуть найденное
# минимальное расстояние. Решение будет проверяться на ряде тестов.
# Считается, что решение прошло тест, если найденное минимальное расстояние
# не больше чем на 4, чем реальное минимальное расстояние.
# Имейте ввиду, что граф может содержать мосты, а так же вершины, удаление
# которых увеличивает компоненту связности графа на единицу.

import numpy as np
from collections import defaultdict
from heapq import *
import time

start_time = time.monotonic()

initial_variants = list()
variants = list()
current_min = None


def find_edges(g):
    G = np.matrix(g)
    N = len(G)
    edges = list()

    for i in range(N):
        for j in range(N):
            if G[(i, j)] != 0:
                edges.append((i, j, G[(i, j)]))

    sorted_edges = sorted(edges, key=lambda x: (x[0], x[2]))
    return sorted_edges


def dijkstra(edges, f, t):
    g = defaultdict(list)
    for l, r, c in edges:
        g[l].append((c, r))

    q, seen, mins = [(0, f, ())], set(), {f: 0}
    while q:
        (cost, v1, path) = heappop(q)
        if v1 not in seen:
            seen.add(v1)
            path = path + tuple([v1])
            if v1 == t:
                return (cost, path)

            for c, v2 in g.get(v1, ()):
                if v2 in seen:
                    continue
                prev = mins.get(v2, None)
                next = cost + c
                if prev is None or next < prev:
                    mins[v2] = next
                    heappush(q, (next, v2, path))

    return float("inf")


def full_sort_fn(v):
    if v[0] < v[1]:
        a = str(v[0]) + str(v[1])
        b = str(v[1]) + str(v[0])
    else:
        a = str(v[1]) + str(v[0])
        b = str(v[0]) + str(v[1])

    return (v[2], int(b) + abs(int(a) - int(b)))


M = list()
N = int(input())

for i in range(N):
    M.append(list(map(int, input().split())))

for i in range(N):
    M[i][i] = 0

edges = find_edges(M)

start = 0
current_min = 0
distance = dict()
for i in range(1, N):
    res = dijkstra(edges, start, i)

    if res != float('inf'):
        current_min += res[0]
        distance[i] = res

current_min *= 2

am = M

# TODO: дополнительная оптимизиция, если число ребёр слишком велико

graph = dict()
for i in range(N):
    graph[i] = list()

for i in range(N):
    for j in range(N):
        if am[i][j] != 0:
            graph[i].append((i, j, am[i][j]))

for key in graph:
    graph[key] = sorted(graph[key], key=full_sort_fn)
    graph[key] = list(map(lambda p: (p[0], p[1]), graph[key]))


def path_len(path):
    global am

    result = 0
    for step in path:
        try:
            weight = M[step[0]][step[1]]
            result += weight
        except:
            pass
    return result


def walk(current, target, prev=None, path=[]):
    global distance
    global current_min
    path.append((prev, current))

    current_path_len = path_len(path)
    if current_path_len >= current_min:
        path.pop()
        return

    visited = set(map(lambda k: k[1], path))
    if current == target:
        if len(visited) == N:
            dist = current_path_len + distance[current][0]
            if dist < current_min:
                current_min = dist
    else:
        step_done = False
        for step in graph[current]:
            if step not in path and step[1] not in visited:
                walk(step[1], target, current, path)
                step_done = True

        if not step_done:
            for step in graph[current]:
                if step not in path:
                    walk(step[1], target, current, path)

    path.pop()
    return


for i in range(1, N):
    walk(start, i)

print(current_min)
