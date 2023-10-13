import numpy as np

def calculate_score(output, label):
    # Tính edit distance
    edit_distance = 0
    for i in range(max(len(output), len(label))):
        if i < len(output):
            pred = output[i]
        else:
            pred = ''
        if i < len(label):
            gt = label[i]
        else:
            gt = ''
        if pred != gt:
            edit_distance += 1
    # Số ký tự trong nhãn
    n = len(label)

    # Nếu output và label giống nhau thì trả về 1
    if edit_distance == 0:
        return 1.0
    # Tính điểm theo công thức
    score = max(0, 1 - (1.5 ** edit_distance) / n)
    return score