import numpy as np
import matplotlib.pyplot as plt

def graphical_method(x1, x2, x2_, root, title):
    plt.plot(x1, x2, label='Series 1')
    plt.plot(x1, x2_, label='Series 2')
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2, x2_')
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.legend()
    plt.grid(True)

    # Highlight the intersection point
    intersection = root
    if intersection is not None:
        plt.plot(intersection[0], intersection[1], 'ro')
        plt.text(intersection[0], intersection[1],
                 f'({intersection[0]:.2f}, {intersection[1]:.2f})', fontsize=12, ha='right')

    plt.show()


def solve_linalg(matrix_A, matrix_B, use_linalg=False):
    try:
        # Print input matrices
        print("Matrix A: \n" + str(matrix_A))
        print("Matrix B: \n" + str(matrix_B))

        # Check if matrix_A is square
        if len(matrix_A) != len(matrix_A[0]):
            raise ValueError("Matrix A must be square.")

        # Print the inverse of matrix A (if it exists)
        inv_A = np.linalg.inv(matrix_A)
        print("Inverse of Matrix A: \n" + str(inv_A))

        # Solve the system of equations
        if use_linalg:
            X = np.linalg.solve(matrix_A, matrix_B)
        else:
            X = inv_A.dot(matrix_B)

        return X

    except np.linalg.LinAlgError as e:
        print(f"An error occurred with the linear algebra operation: {e}")
        return None
    except ValueError as ve:
        print(f"Value Error: {ve}")
        return None
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")
        return None


def calculate_determinant(matrix, use_library=False):
    if use_library:
        return np.linalg.det(matrix)

    if len(matrix) != len(matrix[0]):
        raise ValueError("Matrix must be square")

    if len(matrix) == 2 and len(matrix[0]) == 2:
        determinant = matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
        return determinant

    determinant = 0
    indices = list(range(len(matrix)))

    for fc in indices:
        submatrix = []
        for i in range(1, len(matrix)):
            row = list(matrix[i][:fc]) + list(matrix[i][fc+1:])
            submatrix.append(row)

        sign = (-1) ** fc
        sub_det = calculate_determinant(submatrix)
        determinant += sign * matrix[0][fc] * sub_det

    return determinant

def solve_cramer_rule(A, B, determinant):
    if determinant == 0:
        raise ValueError("Determinant is zero!")
    root = np.zeros(len(A))
    for i in range(len(A)):
        matrix_A = A.copy()
        matrix_A[:, i] = B
        root[i] = calculate_determinant(matrix_A) / determinant
    return root

def solve_naive_gauss(A, B):
    A = A.astype(float)
    B = B.astype(float)
    n = len(B)
    B = B.reshape(-1, 1)
    AB = np.hstack((A, B))
    print(str(AB) + "\n")
    # Gaussian elimination with partial pivoting
    for i in range(n):
        # Pivoting: Swap rows if the diagonal element is zero or near zero
        max_row = np.argmax(np.abs(AB[i:n, i])) + i
        if i != max_row:
            AB[[i, max_row]] = AB[[max_row, i]]
        print(str(AB) + "\n")

        # Ensure the pivot element is non-zero
        if np.isclose(AB[i, i], 0):
            raise ValueError("Matrix is singular or nearly singular")

        # Eliminate elements below the pivot
        for j in range(i + 1, n):
            factor = AB[j, i] / AB[i, i]
            AB[j] -= factor * AB[i]
            print(str(AB) + "\n")

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (AB[i, -1] - np.sum(AB[i, i + 1:n] * x[i + 1:n])) / AB[i, i]

    return x

def test_exercise_2(roots):
    assert np.isclose(2*roots[1] + 5*roots[2], 9), "Equation 1 failed"
    assert np.isclose(2*roots[0] + roots[1] + roots[2], 9), "Equation 2 failed"
    assert np.isclose(3*roots[0] + roots[1], 10), "Equation 3 failed"
    print("All tests passed.")
    
def test_exercise_1_2(roots):
    assert np.isclose(-1.1*roots[0] + 10*roots[1], 120), "Equation 1 failed"
    assert np.isclose(-2*roots[0] + 17.4*roots[1], 174), "Equation 2 failed"
    print("All tests passed.")


def test_exercise_3(roots):
    assert np.isclose(2 * roots[1] + 5 * roots[2], 9), "Equation 1 failed"
    assert np.isclose(2 * roots[0] + roots[1] + roots[2], 9), "Equation 2 failed"
    assert np.isclose(3 * roots[0] + roots[1], 10), "Equation 3 failed"
    print("All tests passed.")

if __name__ == '__main__':
    # Exercise 1 - Number 1
    # 4x1 - 8x2 = -24
    # -x1 + 6x2 = 34
    print("Exercise 1 - Number 1")
    x1 = np.linspace(0, 20)
    x2 = (4 * x1 + 24)/8
    x2_ = (x1 + 34)/6
    root = solve_linalg(np.array([[4, -8], [-1, 6]]), np.array([-24, 34]))
    graphical_method(x1, x2, x2_, root, "Exercise 1 - Number 1")
    print("Therefore the solution of the system of equations based on the intersection is: " + str(root))

    # Exercise 1 - Number 2
    # -1.1x1 + 10x2 = 120
    # -2x1 + 17.4x2 = 174
    print("-------------------------------------")
    print("Exercise 1 - Number 2a")
    x1 = np.linspace(0, 500)
    x2 = (1.1 * x1 + 120)/10
    x2_ = (2 * x1 + 174)/17.4
    root_2 = solve_linalg(np.array([[-1.1, 10], [-2, 17.4]]), np.array([120, 174]))
    graphical_method(x1, x2, x2_, root_2, "Exercise 1 - Number 2a")
    test_equation_1 = -1.1 * root_2[0] + 10 * root_2[1]
    test_equation_2 = -2 * root_2[0] + 17.4 * root_2[1]
    print("Test the solution of the system of equations by substitution:")
    print("Test eq 1: " + str(round(test_equation_1)) + " Test eq 2: " + str(round(test_equation_2)))
    test_exercise_1_2(root_2)
    print("Therefore the solution of the system of equations is correct based on the intersection is: " + str(root_2))
    
    print("-------------------------------------")
    print("Exercise 1 - Number 2c")
    matrix_A = np.array([[1.1, 10], [-2, 17.4]])
    matrix_B = np.array([120, 174])

    print("Determinant of matrix_A:")
    print("Result using library: " + str(calculate_determinant(matrix_A)))  # Without using library
    print("Result without library: " + str(round(calculate_determinant(matrix_A, True), 2)))  # Using library
    
    # Exercise 2
    # 2x2 + 5x3 = 9
    # 2x1 + x2 + x3 = 9
    # 3x1 + x2 = 10
    print("-------------------------------------")
    print("Exercise 2a")
    A = np.array([[0, 2, 5], [2, 1, 1], [3, 1, 0]])
    B = np.array([9, 9, 10])
    A_determinant = calculate_determinant(A)
    print("Matrix A:")
    print(A)
    print("Matrix B:")
    print(B)
    print("Determinant of matrix A: " + str(A_determinant))
    print("-------------------------------------")
    print("Exercise 2b")
    root_3 = []
    root_3 = solve_cramer_rule(A, B, A_determinant)
    print("x1: " + str(root_3[0]) + " x2: " + str(root_3[1]) + " x3: " + str(root_3[2]))
    print("-------------------------------------")
    print("Exercise 2c")
    print("Test result:")
    print("Test eq 1: " + str(2*root_3[1] + 5*root_3[2]))
    print("Test eq 2: " + str(2*root_3[0] + root_3[1] + root_3[2]))
    print("Test eq 3: " + str(3*root_3[0] + root_3[1]))
    test_exercise_2(root_3)

    # Exercise 3
    # 2x2 + 5x3 = 9
    # 2x1 + x2 + x3 = 9
    # 3x1 + x2 = 10
    print("-------------------------------------")
    print("Exercise 3a")
    print("Determinant of matrix A(same as 2a): " + str(A_determinant))
    print("-------------------------------------")
    print("Exercise 3b")
    solution = solve_naive_gauss(A, B)
    print("Solution using naive Gaussian elimination:")
    print(solution)
    
    print("-------------------------------------")
    print("Exercise 3c")
    # Verification by substituting the solution back into the original equations
    print("Verification:")
    print("2 * x2 + 5 * x3 =", 2 * solution[1] + 5 * solution[2])
    print("2 * x1 + x2 + x3 =", 2 * solution[0] + solution[1] + solution[2])
    print("3 * x1 + x2 =", 3 * solution[0] + solution[1])
    test_exercise_3(solution)