"""Python controller"""

from exercise_example import exercise_example
from exercise_8b import exercise_8b
from exercise_8c import exercise_8c
from exercise_8d import exercise_8d1, exercise_8d2
from exercise_8f import exercise_8f
from exercise_8g import exercise_8g


def exercise_all(arguments):
    """Run all exercises"""

    # Get supervisor to take over the world
    timestep = 1e-2

    # Exercise example to show how to run a grid search
    if 'example' in arguments:
        exercise_example(timestep)

    # Exercise 8b - Phase lag + amplitude study
    if '8b' in arguments:
        exercise_8b(timestep)

    # Exercise 8c - Gradient amplitude study
    if '8c' in arguments:
        exercise_8c(timestep)

    # Exercise 8d1 - Turning
    if '8d1' in arguments:
        exercise_8d1(timestep)

    # Exercise 8d2 - Backwards swimming
    if '8d2' in arguments:
        exercise_8d2(timestep)

    # Exercise 8f - Walking
    if '8f' in arguments:
        exercise_8f(timestep)

    # Exercise 8g - Transitions
    if '8g' in arguments:
        exercise_8g(timestep)


if __name__ == '__main__':
    exercise_all(arguments=['8b'])

