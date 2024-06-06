"""
Weeek 4: Liner Regression
"""

# node = hou.pwd()
# geo = node.geometry()
# for i, x in enumerate(x_values):
#     point = geo.createPoint()
#     point.setPosition((x, y_values[i], 0))


def mean(values):
    return sum(values) / float(len(values))


def covariance(x, mean_x, y, mean_y):
    covariance = 0
    for i in range(len(x)):
        covariance += (x[i] - mean_x) * (y[i] - mean_y)

    return covariance


def variance(values, mean_value):
    return sum((x - mean_value) ** 2 for x in values)


def coefitients(x, mean_x, y, mean_y):
    m = covariance(x, mean_x, y, mean_y) / variance(x, mean_x)
    b = mean_y - m * mean_x

    return m, b


def simple_linear_regression(x, slope, intersect):
    """
    y = mx + b
    """

    return slope * x + intersect


x_values = [i for i in range(0, 200)]
y_values = [i for i in range(0, 400, 2)]

mean_x = mean(x_values)
mean_y = mean(y_values)
slope, intersect = coefitients(x_values, mean_x, y_values, mean_y)

new_x = 6
prediction = simple_linear_regression(new_x, slope, intersect)
print(prediction)

