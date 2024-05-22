"""
Find a path from start (3) to the end (2) in 2d array of zeros (path) and ones (wall)
"""

maze = [[1, 1, 1, 1, 1, 2, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 1, 0, 1],
        [1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
        [1, 1, 1, 1, 3, 0, 1, 1, 0, 1]]


def find_start_and_end(matrix):

    start = None
    end = None

    for row_index, row in enumerate(matrix):
        for column_index, column in enumerate(row):
            if column == 3:
                start = (row_index, column_index)
            if column == 2:
                end = (row_index, column_index)

    return start, end


def is_valid_move(matrix, visited, row, column):
    """
    Check if current movement is valid:
        - do not go out of matrix
        - not a wall (value=1)
        - has not been visited
    """

    number_of_rows = len(matrix)
    number_of_columns = len(matrix[0])

    if (0 <= row < number_of_rows) and (0 <= column < number_of_columns) and matrix[row][column] != 1 and not visited[row][column]:
        return True


def find_path(matrix, start, end):

    # All possible directions we can go
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    # Valid path to go from start to end
    paths = []

    # Visited nodes
    visited = [[False for _ in range(len(matrix[0]))] for _ in range(len(matrix))]

    def dfs(row, column):

        if (row, column) == end:
            paths.append((row, column))
            return True

        if is_valid_move(matrix, visited, row, column):
            visited[row][column] = True
            paths.append((row, column))

            for x, y in directions:
                if dfs(row + x, column + y):
                    return True

            paths.pop()

        return False

    start_row, start_column = start
    dfs(start_row, start_column)

    return paths


start, end = find_start_and_end(maze)
solution_paths = find_path(maze, start, end)
print(solution_paths)
