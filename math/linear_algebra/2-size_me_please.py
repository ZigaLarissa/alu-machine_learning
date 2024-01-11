def matrix_shape(matrix):
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0] if matrix else None
    return shape
