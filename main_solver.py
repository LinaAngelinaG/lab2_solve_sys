import numpy as np


def sort_columns_by_absmax(matrix):
    """
    Sort the columns of the matrix in descending order based on the absolute
    value of their first element. Move all zero columns to the left end of
    the matrix.
    """
    k = matrix.shape[1] - 1
    while k > 0:
        ind = np.argmax(np.abs(matrix[0, 1:k + 1]))
        matrix[:, [ind, k]] = matrix[:, [k, ind]]
        k -= 1
    nulls = np.count_nonzero(matrix[0] == 0)
    if nulls > 0:
        matrix[:, [0, nulls]] = matrix[:, [nulls, 0]]
    return matrix


def divide_arr(array, step):
    """
    Return an array of divisors obtained by dividing each element in the array
    by the given divider. If the divider is negative, return the absolute value
    of the quotient for negative elements and the negated quotient for positive
    elements.
    """
    if step < 0:
        return [-1 * (-elem // step) if elem > 0 else abs(elem // abs(step))
                for elem in array]
    else:
        return array // step


def gaussian_elimination(matrix):
    """
    Perform Gaussian elimination on the given matrix to solve a system of linear
    equations. Return the reduced row echelon form of the matrix.
    """
    for i in range(np.shape(matrix)[0]):
        while not np.all(matrix[i][i + 1:] == 0):
            matrix[i:, i:] = sort_columns_by_absmax(matrix[i:, i:])
            dividers = divide_arr(matrix[i][i + 1:], matrix[i][i])
            for j, div in enumerate(dividers):
                matrix[:, (i + 1) + j] -= matrix[:, i] * div
    return matrix


def solve_system(matrix, vector):
    """
    Solve the system of linear equations Ax = b for x, where A is the given
    matrix and b is the given vector. Return a tuple (success, solution), where
    success is a boolean indicating whether the system was solved in whole
    numbers, and solution is the particular solution x_0 if success is True or
    the general solution if success is False.
    """
    matrix, vector = np.array(matrix), np.array(vector)
    n, m = len(matrix[0]), len(matrix)

    matrix_e = np.concatenate((matrix, np.identity(n, dtype=int)), axis=0)
    matrix_c = gaussian_elimination(matrix_e)

    rank = get_matrix_rank(matrix_c)

    return finish_solution(rank, matrix_c, np.concatenate([[[(-1) * x[0]] for x in vector], np.zeros((n, 1), dtype=int)], axis=0), m)


def get_matrix_rank(matrix):
    """Return the rank of the matrix."""
    counter = 0
    for i in range(np.shape(matrix)[0] - np.shape(matrix)[1]):
        if matrix[i][i] <= 0:
            break
        counter = i
    return counter + 1


def finish_solution(rank, matrix_c, a_values, m):
    counter = 0
    while counter < rank and matrix_c[counter][counter] != 0:
        a_values[:, 0] = a_values[:, 0] - a_values[counter][0] // matrix_c[counter][counter] * matrix_c[:, counter]
        counter += 1

    if np.all(a_values[:m] == 0):
        solution = a_values[m:, :]
        return True, solution
    else:
        general_solution = a_values[m:, :]
        return False, general_solution
