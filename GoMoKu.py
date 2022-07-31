from graphics import *
from math import *
import numpy as np


def ai():
    cut_count = 0
    search_count = 0
    t = time.time()
    alpha_beta(True, DEPTH, -99999999, 99999999)
    print(f' running time {time.time()-t:.4F} s')
    return next_point[0], next_point[1]

# threat seqence
shape_score = [(500, (0, 1, 1, 0, 0)),
               (500, (0, 0, 1, 1, 0)),
               (2000, (1, 1, 0, 1, 0)),
               (2000, (0, 1, 0, 1, 1)),
               (5000, (0, 0, 1, 1, 1)),
               (5000, (1, 1, 1, 0, 0)),
               (50000, (0, 1, 1, 1, 0)),
               (50000, (0, 1, 0, 1, 1, 0)),
               (50000, (0, 1, 1, 0, 1, 0)),
               (50000, (1, 1, 1, 0, 1)),
               (50000, (1, 1, 0, 1, 1)),
               (50000, (1, 0, 1, 1, 1)),
               (50000, (1, 1, 1, 1, 0)),
               (50000, (0, 1, 1, 1, 1)),
               (200000, (0, 1, 1, 1, 1, 0)),
               (99999999, (1, 1, 1, 1, 1))]


# alpha beta pruning
def alpha_beta(is_ai, depth, alpha, beta):
    # check if game is over or reach the bottom of the tree
    if ifOver(list1) or ifOver(list2) or depth == 0:
        return evaluation(is_ai)

    blank_list = list(set(list_all).difference(set(list3)))
    last_sort(blank_list)   # herustic, which give the last steps priorty, speed up the search
    # loop over every step
    for next_step in blank_list:

        global search_count

        # if there are no neighbour it is likely have less chance to win
        if not neighbour(next_step):
            continue

        if is_ai:
            list1.append(next_step)
        else:
            list2.append(next_step)
        list3.append(next_step)

        value = -alpha_beta(not is_ai, depth - 1, -beta, -alpha)
        if is_ai:
            list1.remove(next_step)
        else:
            list2.remove(next_step)
        list3.remove(next_step)

        if value > alpha:

            print("current val: " + str(value) + "alpha:" + str(alpha) + "beta:" + str(beta))
            if depth == DEPTH:
                next_point[0] = next_step[0]
                next_point[1] = next_step[1]
            # pruning
            if value >= beta:
                return beta
            alpha = value

    return alpha


#  The last one' neighbour heuristic
def last_sort(blank_list):
    last_po = list3[-1]
    for item in blank_list:
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                if (last_po[0] + i, last_po[1] + j) in blank_list:
                    blank_list.remove((last_po[0] + i, last_po[1] + j))
                    blank_list.insert(0, (last_po[0] + i, last_po[1] + j))


def neighbour(po):
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            if (po[0] + i, po[1]+j) in list3:
                return True
    return False


# evaluation function
def evaluation(is_ai):
    total_score = 0

    if is_ai:
        my_list = list1
        enemy_list = list2
    else:
        my_list = list2
        enemy_list = list1

    # AI score
    score_all_arr = []  # store the interscation pos
    my_score = 0
    for po in my_list:
        m = po[0]
        n = po[1]
        my_score += cal_score(m, n, 0, 1, enemy_list, my_list, score_all_arr)
        my_score += cal_score(m, n, 1, 0, enemy_list, my_list, score_all_arr)
        my_score += cal_score(m, n, 1, 1, enemy_list, my_list, score_all_arr)
        my_score += cal_score(m, n, -1, 1, enemy_list, my_list, score_all_arr)

    # Human socre
    score_all_arr_enemy = []
    enemy_score = 0
    for po in enemy_list:
        m = po[0]
        n = po[1]
        enemy_score += cal_score(m, n, 0, 1, my_list, enemy_list, score_all_arr_enemy)
        enemy_score += cal_score(m, n, 1, 0, my_list, enemy_list, score_all_arr_enemy)
        enemy_score += cal_score(m, n, 1, 1, my_list, enemy_list, score_all_arr_enemy)
        enemy_score += cal_score(m, n, -1, 1, my_list, enemy_list, score_all_arr_enemy)

    total_score = my_score - enemy_score*0.01

    return total_score


# using x and y to get the diagonal direaction
def cal_score(m, n, x, y, enemy_list, my_list, score_all_arr):
    add_score = 0  # if there exist a double threat, give bonus

    max_score_shape = (0, None)

    for item in score_all_arr:
        for po in item[1]:
            if m == po[0] and n == po[1] and x == item[2][0] and y == item[2][1]:
                return 0

    for bias in range(-5, 1):
        pos = []
        for i in range(0, 6):
            if (m + (i + bias) * x, n + (i + bias) * y) in enemy_list:
                pos.append(2)
            elif (m + (i + bias) * x, n + (i + bias) * y) in my_list:
                pos.append(1)
            else:
                pos.append(0)
        tmp_shap5 = (pos[0], pos[1], pos[2], pos[3], pos[4])
        tmp_shap6 = (pos[0], pos[1], pos[2], pos[3], pos[4], pos[5])

        for (score, shape) in shape_score:
            if tmp_shap5 == shape or tmp_shap6 == shape:
                if score > max_score_shape[0]:
                    max_score_shape = (score, ((m + (0+bias) * x, n + (0+bias) * y),
                                               (m + (1+bias) * x, n + (1+bias) * y),
                                               (m + (2+bias) * x, n + (2+bias) * y),
                                               (m + (3+bias) * x, n + (3+bias) * y),
                                               (m + (4+bias) * x, n + (4+bias) * y)), (x, y))

    # check if there exist a double threat
    if max_score_shape[1] is not None:
        for item in score_all_arr:
            for po1 in item[1]:
                for po2 in max_score_shape[1]:
                    if po1 == po2 and max_score_shape[0] > 10 and item[0] > 10:
                        add_score += item[0] + max_score_shape[0]

        score_all_arr.append(max_score_shape)

    return add_score + max_score_shape[0]


def ifOver(list):
    for m in range(column):
        for n in range(row):

            if n < row - 4 and (m, n) in list and (m, n + 1) in list and (m, n + 2) in list and (
                    m, n + 3) in list and (m, n + 4) in list:
                return True
            elif m < row - 4 and (m, n) in list and (m + 1, n) in list and (m + 2, n) in list and (
                        m + 3, n) in list and (m + 4, n) in list:
                return True
            elif m < row - 4 and n < row - 4 and (m, n) in list and (m + 1, n + 1) in list and (
                        m + 2, n + 2) in list and (m + 3, n + 3) in list and (m + 4, n + 4) in list:
                return True
            elif m < row - 4 and n > 3 and (m, n) in list and (m + 1, n - 1) in list and (
                        m + 2, n - 2) in list and (m + 3, n - 3) in list and (m + 4, n - 4) in list:
                return True
    return False


def window():
    win = GraphWin("GoMoKu", width * column, width * row)
    win.setBackground("white")
    i1 = 0

    while i1 <= width * column:
        l = Line(Point(i1, 0), Point(i1, width * column))
        l.draw(win)
        i1 = i1 + width
    i2 = 0

    while i2 <= width * row:
        l = Line(Point(0, i2), Point(width * row, i2))
        l.draw(win)
        i2 = i2 + width
    return win


def main():
    win = window()

    for i in range(column+1):
        for j in range(row+1):
            list_all.append((i, j))

    change = 0
    g = 0
    m = 0
    n = 0

    while g == 0:

        if change == 0:
            pos = (7,7)
            list1.append(pos)
            list3.append(pos)

            piece = Circle(Point(width * pos[0], width * pos[1]), 16)
            piece.setFill('black')
            piece.draw(win)

            if ifOver(list1):
                message = Text(Point(100, 100), "black win.")
                message.draw(win)
                g = 1
            change = change + 1

        if change % 2 == 0:
            pos = ai()

            list1.append(pos)
            list3.append(pos)

            piece = Circle(Point(width * pos[0], width * pos[1]), 16)
            piece.setFill('black')
            piece.draw(win)

            if ifOver(list1):
                message = Text(Point(100, 100), "black win.")
                message.draw(win)
                g = 1
            change = change + 1

        else:
            p2 = win.getMouse()
            if not ((round((p2.getX()) / width), round((p2.getY()) / width)) in list3):

                a2 = round((p2.getX()) / width)
                b2 = round((p2.getY()) / width)
                list2.append((a2, b2))
                list3.append((a2, b2))

                piece = Circle(Point(width * a2, width * b2), 16)
                piece.setFill('red')
                piece.draw(win)
                if ifOver(list2):
                    message = Text(Point(100, 100), "red win.")
                    message.draw(win)
                    g = 1

                change = change + 1

    win.getMouse()
    win.close()

# init window
width = 60
column = 15
row = 15

# init search env
list1 = []  # AI stone
list2 = []  # human stone
list3 = []  # occupied position

list_all = []  # remain position
next_point = [0, 0]  # next move

DEPTH = 3  #  search depth


main()
