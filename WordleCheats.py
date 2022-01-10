import json, re, colorsys, os, string
import random

import cv2 as cv
from PIL import Image, ImageFilter, ImageChops, ImageOps, ImageGrab
import matplotlib.pyplot as plt
import numpy as np
import math

# filename = "extra_wordset.json"
filename = "scrabble2019.json"
with open(filename) as json_file:
    wordset = json.load(json_file)

wordset = [w.lower() for w in wordset]

# letter_values = [{'s': 0.120016, 'a': 0.074181, 'b': 0.069959, 'c': 0.06882, 't': 0.067212, 'p': 0.059304, 'm': 0.051799, 'd': 0.046572, 'g': 0.04617, 'r': 0.044763, 'l': 0.041614, 'f': 0.03833, 'h': 0.033773, 'w': 0.03317, 'k': 0.031294, 'n': 0.026603, 'e': 0.02399, 'u': 0.022583, 'o': 0.020907, 'j': 0.018093, 'i': 0.016954, 'v': 0.016418, 'y': 0.013067, 'z': 0.006902, 'q': 0.005495, 'x': 0.00201}, {'a': 0.182269, 'o': 0.140186, 'e': 0.121557, 'i': 0.102258, 'u': 0.091939, 'r': 0.070897, 'l': 0.055418, 'h': 0.049521, 'n': 0.033572, 't': 0.021041, 'y': 0.018361, 'p': 0.016351, 'c': 0.014876, 'm': 0.014139, 'w': 0.011928, 's': 0.009717, 'd': 0.009113, 'k': 0.00784, 'g': 0.006567, 'b': 0.006366, 'v': 0.004758, 'x': 0.004155, 'z': 0.00268, 'f': 0.002144, 'j': 0.001273, 'q': 0.001072}, {'a': 0.101186, 'r': 0.096227, 'i': 0.082222, 'n': 0.077799, 'o': 0.075923, 'e': 0.068619, 'l': 0.067145, 'u': 0.052804, 't': 0.047041, 's': 0.041346, 'm': 0.039737, 'd': 0.031696, 'c': 0.029418, 'b': 0.027005, 'g': 0.026, 'p': 0.024861, 'k': 0.0195, 'w': 0.015882, 'v': 0.01568, 'f': 0.012933, 'y': 0.012531, 'h': 0.012464, 'z': 0.008309, 'x': 0.008175, 'j': 0.004021, 'q': 0.001474}, {'e': 0.146485, 'a': 0.111238, 'i': 0.077732, 't': 0.069423, 'n': 0.063861, 'l': 0.058835, 'r': 0.054145, 'o': 0.042954, 'u': 0.04215, 'd': 0.03967, 's': 0.038531, 'k': 0.034175, 'c': 0.032433, 'm': 0.032299, 'g': 0.031964, 'p': 0.028077, 'b': 0.019567, 'h': 0.018093, 'f': 0.014005, 'y': 0.011995, 'v': 0.011191, 'w': 0.010923, 'z': 0.006701, 'j': 0.002345, 'x': 0.001005, 'q': 0.000201}, {'s': 0.335723, 'e': 0.097836, 'y': 0.092207, 'a': 0.071299, 'n': 0.050459, 't': 0.049387, 'r': 0.049052, 'l': 0.03967, 'd': 0.030691, 'o': 0.028346, 'h': 0.027675, 'i': 0.026603, 'k': 0.021979, 'm': 0.015613, 'c': 0.012598, 'p': 0.011861, 'g': 0.00918, 'x': 0.006433, 'u': 0.006299, 'f': 0.005361, 'b': 0.005026, 'w': 0.004021, 'z': 0.002144, 'v': 0.000402, 'q': 6.7e-05, 'j': 6.7e-05}]
letter_values = [{'s': 0.120016, 'a': 0.074181, 'b': 0.069959, 'c': 0.06882, 't': 0.067212, 'p': 0.059304, 'm': 0.051799, 'd': 0.046572, 'g': 0.04617, 'r': 0.044763, 'l': 0.041614, 'f': 0.03833, 'h': 0.033773, 'w': 0.03317, 'k': 0.031294, 'n': 0.026603, 'e': 0.02399, 'u': 0.022583, 'o': 0.020907, 'j': 0.018093, 'i': 0.016954, 'v': 0.016418, 'y': 0.013067, 'z': 0.006902, 'q': 0.005495, 'x': 0.00201}, {'a': 0.182269, 'o': 0.140186, 'e': 0.121557, 'i': 0.102258, 'u': 0.091939, 'r': 0.070897, 'l': 0.055418, 'h': 0.049521, 'n': 0.033572, 't': 0.021041, 'y': 0.018361, 'p': 0.016351, 'c': 0.014876, 'm': 0.014139, 'w': 0.011928, 's': 0.009717, 'd': 0.009113, 'k': 0.00784, 'g': 0.006567, 'b': 0.006366, 'v': 0.004758, 'x': 0.004155, 'z': 0.00268, 'f': 0.002144, 'j': 0.001273, 'q': 0.001072}, {'a': 0.101186, 'r': 0.096227, 'i': 0.082222, 'n': 0.077799, 'o': 0.075923, 'e': 0.068619, 'l': 0.067145, 'u': 0.052804, 't': 0.047041, 's': 0.041346, 'm': 0.039737, 'd': 0.031696, 'c': 0.029418, 'b': 0.027005, 'g': 0.026, 'p': 0.024861, 'k': 0.0195, 'w': 0.015882, 'v': 0.01568, 'f': 0.012933, 'y': 0.012531, 'h': 0.012464, 'z': 0.008309, 'x': 0.008175, 'j': 0.004021, 'q': 0.001474}, {'e': 0.146485, 'a': 0.111238, 'i': 0.077732, 't': 0.069423, 'n': 0.063861, 'l': 0.058835, 'r': 0.054145, 'o': 0.042954, 'u': 0.04215, 'd': 0.03967, 's': 0.038531, 'k': 0.034175, 'c': 0.032433, 'm': 0.032299, 'g': 0.031964, 'p': 0.028077, 'b': 0.019567, 'h': 0.018093, 'f': 0.014005, 'y': 0.011995, 'v': 0.011191, 'w': 0.010923, 'z': 0.006701, 'j': 0.002345, 'x': 0.001005, 'q': 0.000201}, {'s': 0.335723, 'e': 0.097836, 'y': 0.092207, 'a': 0.071299, 'n': 0.050459, 't': 0.049387, 'r': 0.049052, 'l': 0.03967, 'd': 0.030691, 'o': 0.028346, 'h': 0.027675, 'i': 0.026603, 'k': 0.021979, 'm': 0.015613, 'c': 0.012598, 'p': 0.011861, 'g': 0.00918, 'x': 0.006433, 'u': 0.006299, 'f': 0.005361, 'b': 0.005026, 'w': 0.004021, 'z': 0.002144, 'v': 0.000402, 'q': 6.7e-05, 'j': 6.7e-05}]


def dist(rgb, col=(118, 103, 80), v1=15, v2=15, v3=15):  # default col is average grid color
    # h, s, v = (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
    h, s, v = rgb
    if abs(h - col[0]) < v1:
        if abs(s - col[1]) < v2:
            if abs(v - col[2]) < v3:
                return True
    return False


def simplify(inp, inp2, weighted=False):
    out_arr = []
    if weighted:
        arr = [inp[0] * inp2[0]]
    else:
        arr = [inp[0]]
    arr2 = [inp2[0]]
    if weighted:
        for i in range(1, len(inp)):
            if inp[i] - inp[i - 1] > 1:
                out = sum(arr) / sum(arr2)
                out_arr.append(out)
                arr = [inp[i] * inp2[i]]
                arr2 = [inp2[i]]
            else:
                arr.append(inp[i] * inp2[i])
                arr2.append(inp2[i])
    else:
        for i in range(1, len(inp)):
            if inp[i] - inp[i - 1] > 1:
                out = sum(arr) / len(arr)
                out_arr.append(out)
                arr = [inp[i]]
            else:
                arr.append(inp[i])
    return out_arr


def solve2(inp):
    # arr = [a - inp[0] for a in inp][1:]

    inp = inp[1:-1]  # the left and right sides are empty space and don't have grid relevance
    offset = inp[0]

    # arr = list(map(lambda a: a - offset, inp))[1:]
    arr = [a - offset for a in inp]
    m_arr = []
    # for i in range(1, len(arr)):
    #     m_arr.append(arr[i] - arr[i - 1])
    m_arr = list(map(lambda i: arr[i] - arr[i - 1], range(1, len(arr))))

    for i in range(1, len(arr)-1):
        diff = m_arr[i - 1] - m_arr[i]
        if diff > 2:
            for r in range(2, 6):
                if r + 0.1 > m_arr[i - 1] / m_arr[i] > r - 0.1:
                    m_arr[i - 1] = m_arr[i - 1] / r

    if len(m_arr) > 1:
        m_arr.sort()
        m_arr2 = [m_arr[1]]
        for i in range(1, len(m_arr)):
            diff = m_arr[i] - m_arr[i - 1]
            if diff > 1:
                break
            m_arr2.append(m_arr[i])
    else:
        m_arr2 = m_arr
    m = sum(m_arr2) / len(m_arr2)
    n = round(arr[-1] / m)
    m = arr[-1] / n
    while offset > m:
        offset -= m
    return m, offset


def plotty(xarr, yarr, xlim, ylim):
    fig = plt.figure()
    host = fig.add_subplot()
    xarr = [[a[0], xlim] for a in xarr]
    yarr = [[a[0], ylim] for a in yarr]
    host.plot(xarr)
    fig2 = plt.figure()
    part = fig2.add_subplot()
    part.plot(yarr)
    plt.show()


def get_grid(inp, val=1):
    w, h = inp.size
    inp = inp.resize((round(w / val), round(h / val)), Image.NEAREST)
    w, h = inp.size
    img1 = inp.convert("HSV").copy()
    img2 = inp.convert("L").copy()

    iml = img1.load()
    il2 = img2.load()
    col = (18, 18, 19)
    col = colorsys.rgb_to_hsv(col[0] / 255, col[1] / 255, col[2] / 255)
    col = [int(a * 255) for a in col]
    for x in range(w):
        for y in range(h):
            rgb = iml[x, y]
            if dist(rgb, col):
                il2[x, y] = 255
            else:
                il2[x, y] = 0

    w, h = img2.size
    x_arr = []
    y_arr = []
    for x in range(w):
        count = 0
        for y in range(h):
            if il2[x, y] == 255:
                count += 1
        x_arr.append([count, x])

    for y in range(h):
        count = 0
        for x in range(w):
            if il2[x, y] == 255:
                count += 1
        y_arr.append([count, y])

    x_lim = np.percentile([*(a[0] for a in x_arr)], 90)  # to trim unwanted noise
    y_lim = np.percentile([*(a[0] for a in y_arr)], 90)

    # plotty(x_arr, y_arr, x_lim, y_lim)

    x_arr = [*filter(lambda a: a[0] >= x_lim, x_arr)]
    y_arr = [*filter(lambda a: a[0] >= y_lim, y_arr)]

    nx_arr = [*(a[0] for a in x_arr)]
    x_arr = [*(a[1] for a in x_arr)]

    ny_arr = [*(a[0] for a in y_arr)]
    y_arr = [*(a[1] for a in y_arr)]

    xm, xb = solve2(simplify(x_arr, nx_arr))
    ym, yb = solve2(simplify(y_arr, ny_arr))

    if abs(xm - ym) > 0.5:
        if xm < ym:
            ym = xm
        else:
            xm = ym

    xout_arr = []
    for i in range(math.ceil(inp.size[0] / xm)):
        result = round((xm * i + xb) * val)
        if result > inp.size[0]:
            break
        else:
            xout_arr.append(result)

    yout_arr = []
    for i in range(math.ceil(inp.size[1] / ym)):
        result = round((ym * i + yb) * val)
        if result > inp.size[1]:
            break
        else:
            yout_arr.append(result)
    return xout_arr, yout_arr


def show_board(board):
    for y in range(len(board[0])):
        for x in range(len(board)):
            print(board[x][y], end="")
        print("")


def show_grid(img, xarr, yarr):
    img2 = img.copy()  # to prevent editing the original image
    il = img2.load()
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            if x in xarr or y in yarr:
                il[x, y] = (0, 255, 255)
    img2.show()


def normalize(img, debug=False):
    # img = Image.eval(img, (lambda x: (x > 200) * x))
    w, h = img.size
    il = img.load()
    l, u, d, r = w, w, 0, 0
    for x in range(w):
        for y in range(h):
            if il[x, y] >= 190:
                if x < l:
                    l = x
                if y < u:
                    u = y
                if x > r:
                    r = x
                if y > d:
                    d = y
    img = img.crop((l, u, r, d))
    img = img.resize((25, 25), Image.BICUBIC)
    img = Image.eval(img, (lambda a: (a > 190) * 255))
    if debug:
        img.show()
    return img


def sieve(img, letters):  # letter matching similar can be tricky...
    if "Q" in letters and "O" in letters:
        img = img.crop((13, 19, 25, 25))
    if "G" in letters and ("Q" in letters or "O" in letters):
        img = img.crop((20, 10, 25, 20))
    if "E" in letters and "F" in letters:
        img = img.crop((0, 16, 25, 25))
    if "E" in letters and "L" in letters:
        img = img.crop((0, 0, 25, 20))
    if "B" in letters and "D" in letters:
        img = img.crop((0, 8, 25, 17))
    if "R" in letters and "P" in letters:
        img = img.crop((20, 20, 25, 25))
    if "L" in letters and "U" in letters:
        img = img.crop((20, 0, 25, 25))
    if "O" in letters and "D" in letters:
        img = img.crop((0, 0, 5, 5))
    if "H" in letters and "R" in letters:
        img = img.crop((0, 0, 25, 10))
    if "H" in letters and "U" in letters:
        img = img.crop((5, 8, 20, 16))
    return img


def get_sim(img, img2, debug=False):  # this one is currently used, above and below are ignored
    img2 = np.asarray(img2)
    res = cv.matchTemplate(img, img2, cv.TM_SQDIFF_NORMED)
    # print(cv.minMaxLoc(res))
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    return min_val


def most_similar2(im, im_list, diff=False, debug=False):
    let_arr = ["Q", "O", "G", "E", "F", "L", "B", "D", "R", "P", "U", "H"]
    ima = np.asarray(im)
    sim_arr = []
    o_arr = [*zip(map(lambda a: get_sim(ima, a, debug), im_list), list(string.ascii_uppercase))]
    o_arr.sort()
    letters = [o_arr[0][1], o_arr[1][1]]  # to make doubly sure it's what it's supposed to be, the matching can be otherwise drunk with similar characters
    if letters[0] in let_arr or letters[1] in let_arr:
        o_arr = [*zip(map(lambda a: (get_sim(np.asarray(sieve(im, letters)), sieve(a, letters))), im_list), list(string.ascii_uppercase))]
        o_arr = list(filter(lambda a: a[1] in letters, o_arr))
        o_arr.sort()
    num = ord(o_arr[0][1]) - 65
    return num


def check_col(img, col):
    if col == "yellow":
        col = (181, 159, 59)
    if col == "green":
        col = (83, 141, 78)
    if col == "grey":
        col = (58, 58, 60)
    if col == "black":
        col = (18, 18, 19)
    im_iter = img.getdata()
    col = colorsys.rgb_to_hsv(col[0] / 255, col[1] / 255, col[2] / 255)
    col = [int(a * 255) for a in col]
    # col = list(map(lambda a: int(a * 255), col))
    count = sum(map(lambda i: dist(i, col), im_iter))
    perc = count / (img.size[0] * img.size[1])
    return perc


def get_board(xarr, yarr, img):
    xarr = xarr[:-1]
    yarr = yarr[:-1]
    let_dir = "wordle/letters/"
    w, h = img.size
    cube = round(xarr[-1] - xarr[1]) / (len(xarr) - 1)
    letter_files = os.listdir(let_dir)
    im_arr = list(map(lambda a: Image.open(let_dir + a).convert("L"), letter_files))

    char_arr = []
    for x in xarr:
        char_arr2 = []
        for y in yarr:
            char_arr2.append("_")
        char_arr.append(char_arr2)

    img = img.convert("HSV")

    wrong_arr = [[], [], [], [], []]
    req_arr = []
    right_arr = ["", "", "", "", ""]
    not_arr = []

    for ex in xarr:
        xp = xarr.index(ex)
        for ey in yarr:
            yp = yarr.index(ey)
            imc = img.crop((ex, ey, ex + cube, ey + cube))
            w, h = imc.size

            black_perc = check_col(imc, "black")

            if black_perc < 0.2:

                im = normalize(imc.convert("L"))

                # im.save("wordle/debug/" + str(xp) + str(yp) + ".png")

                num = most_similar2(im, im_arr)

                yellow_perc = check_col(imc, "yellow")

                green_perc = check_col(imc, "green")

                char = chr(65 + num)

                if yellow_perc > 0.2:
                    char_arr[xp][yp] = char
                    req_arr = {*req_arr, char.lower()}
                    wrong_arr[xp] = list({*wrong_arr[xp], char.lower()})
                elif green_perc > 0.2:
                    char_arr[xp][yp] = char
                    right_arr[xp] = char.lower()
                    req_arr = {*req_arr, char.lower()}
                else:
                    char_arr[xp][yp] = char.lower()
                    wrong_arr[xp] = list({*wrong_arr[xp], char.lower()})
                    not_arr.append(char.lower())
                #     # print(xarr.index(ex), x_ind, yarr.index(ey), y_ind)
                #     player_squares.append((xp, yp, chr(65 + num)))
                #     char_arr[xp][yp] = chr(65 + num)
                #     # char_arr2.append(chr(65 + num))
            else:
                continue

    return char_arr, list(req_arr), right_arr, wrong_arr, not_arr


def word_value(word):
    # word = "".join(list({*word}))
    vals = []
    for i in range(5):
        vals.append(letter_values[i][word[i]])
    lets = list({*word})
    if len(lets) == 5:
        return sum(vals)
    val = 0
    for l in lets:
        arr = []
        for w in range(5):
            if word[w] == l:
                arr.append(vals[w])
        val += min(arr)
    return val


def get_moves(req_arr, right_arr, wrong_arr, not_arr):
    alph = "abcdefghijklmnopqrstuvwxyz"

    arr = []  # arr is building the regex per each letter position...
    for i in range(5):
        neg = wrong_arr[i]
        out = "[" + alph + "]"
        for n in neg:
            out = out.replace(n, "")
        for n in not_arr:
            out = out.replace(n, "")
        arr.append(out)

    for i in range(5):
        if right_arr[i] != "":
            arr[i] = right_arr[i]

    reg = "".join(arr) + "\\b"
    # print(reg)
    r1 = re.compile(reg)
    out = list(filter(r1.match, wordset))
    for l in req_arr:
        out = list(filter(lambda a: l in a, out))
    out.sort(key=word_value, reverse=True)
    return out


def return_sim_word():
    reg = "[a-z]{5}\\b"
    r1 = re.compile(reg)
    out = list(filter(r1.match, wordset))
    return random.choice(out)


def wordle_simulate(final, input, board=None):
    if board is None:
        char_arr = []
        for x in range(5):
            char_arr2 = []
            for y in range(6):
                char_arr2.append("_")
            char_arr.append(char_arr2)
        board = char_arr
        return board, [], ['', '', '', '', ''], [[], [], [], [], []], []
    wrong_arr = [[], [], [], [], []]
    req_arr = []
    right_arr = ["", "", "", "", ""]
    not_arr = []
    for y in range(6):
        for x in range(5):
            let = board[x][y]
            if let == "_":
                board[x][y] = input[x]

    for y in range(6):
        for x in range(5):
            let = board[x][y].lower()
            valid = final[x].lower()
            if let == valid:
                board[x][y] = char
                right_arr[x] = char.lower()
                req_arr = {*req_arr, char.lower()}
            elif char in final:
                board[x][y] = char
                req_arr = {*req_arr, char.lower()}
                wrong_arr[x] = list({*wrong_arr[xp], char.lower()})
            else:
                board[x][y] = char.lower()
                wrong_arr[x] = list({*wrong_arr[xp], char.lower()})
                not_arr.append(char.lower())




# img = Image.open("wordle/clip2.png")
img = ImageGrab.grabclipboard()  # use win + shift + s to grab wordle board to clipboard, then run script
# img.show()


xarr, yarr = get_grid(img)

# show_grid(img, xarr, yarr)
# plotty(xarr, yarr)

board, req_arr, right_arr, wrong_arr, not_arr = get_board(xarr, yarr, img)

# word = return_sim_word()
#
# board, req_arr, right_arr, wrong_arr, not_arr = wordle_simulate(word)


show_board(board)

print(board[0][1])

if board[0][1] == "_":
    if len(req_arr) <= 2:
        for i in req_arr:
            not_arr = list({*not_arr, i})
        req_arr = []
        for i in range(len(right_arr)):
            if right_arr[i] != '':
                wrong_arr[i].append(right_arr[i])
                right_arr[i] = ''


print(req_arr, right_arr, wrong_arr, not_arr)

words = get_moves(req_arr, right_arr, wrong_arr, not_arr)

if len(words) > 10 * 4:
    for i in range(0, 10*4, 10):
        out = ", ".join(words[i:i + 10])
        print(out)
else:
    for i in range(0, math.ceil(len(words) / 10), 10):
        out = ", ".join(words[i:i + 10])
        print(out)






