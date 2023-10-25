
def calculate_score(output, label):
    # Bước 1: Tính edit distance
    d = edit_distance(output, label)
    # Bước 2: Nếu edit distance bằng 0, trả về điểm là 1, ngược lại áp dụng công thức
    if d == 0:
        score = 1
    else:
        # Tính độ dài của xâu label
        n = len(label)
        # Tính điểm theo công thức
        score = max(0, 1 - (1.5 ** d) / n)
    return score

def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    # Tạo một ma trận để lưu kết quả tính toán edit distance
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            # Nếu một trong hai xâu là rỗng, edit distance bằng độ dài xâu còn lại
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            # Nếu ký tự cuối cùng giống nhau, không cần thêm bước chỉnh sửa
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            # Nếu ký tự cuối cùng khác nhau, cần thêm một bước chỉnh sửa
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],      # Xóa
                                  dp[i][j - 1],      # Chèn
                                  dp[i - 1][j - 1])   # Thay thế
    
    return dp[m][n]