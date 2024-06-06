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


def dfs(matrix, visited, paths, directions, start_row, start_column, end_row, end_column):

    if (start_row, start_column) == (end_row, end_column):
        paths.append((start_row, start_column))
        return True

    if is_valid_move(matrix, visited, start_row, start_column):
        visited[start_row][start_column] = True
        paths.append((start_row, start_column))

        for x, y in directions:
            if dfs(matrix, visited, paths, directions, start_row + x, start_column + y, end_row, end_column):
                return True

        paths.pop()

        return False


def bfs(matrix, visited, paths, directions, start_row, start_column, end_row, end_column):

    queue = [(start_row, start_column, [])]

    while queue:
        current_row, current_column, current_path = queue.pop(0)

        if (current_row, current_column) == (end_row, end_column):
            current_path.append((current_row, current_column))
            paths.extend(current_path)
            return True

        if is_valid_move(matrix, visited, current_row, current_column):
            visited[current_row][current_column] = True
            current_path.append((current_row, current_column))

            for x, y in directions:
                next_row, next_column = current_row + x, current_column + y
                queue.append((next_row, next_column, current_path.copy()))

    return False


def find_path(matrix):

    start, end = find_start_and_end(maze)
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # All possible directions we can go
    paths = []  # Valid path to go from start to end
    visited = [[False for _ in range(len(matrix[0]))] for _ in range(len(matrix))]  # Visited nodes base matrix

    start_row, start_column = start
    end_row, end_column = end

    # Find path with BFS and DFS algorithms
    dfs(matrix, visited, paths, directions, start_row, start_column, end_row, end_column)
    # bfs(matrix, visited, paths, directions, start_row, start_column, end_row, end_column)

    return paths


solution_paths = find_path(maze)
print(solution_paths)
