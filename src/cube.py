from src.environment import Environment
import numpy as np

# This class represents the Rubik's Cube environment.
class Cube(Environment):
    def __init__(self, size):
        self.size = size
        self.cube = self.initialize_clean_cube()

    # This method represents the action of turning a slice of the cube
    def turn_slice(self, axis, index):
        match axis:
            case Axis.X:
                self.cube[index, :, :] = np.rot90(self.cube[index, :, :])
                for block in self.cube[index, :, :].flatten():
                    block.rotate_x()
            case Axis.Y:
                self.cube[:, index, :] = np.rot90(self.cube[:, index, :])
                for block in self.cube[:, index, :].flatten():
                    block.rotate_y()
            case Axis.Z:
                self.cube[:, :, index] = np.rot90(self.cube[:, :, index])
                for block in self.cube[:, :, index].flatten():
                    block.rotate_z()
            case _:
                raise Exception("Unrecognized plane {0}".format(axis))

        return self.get_state(), self.get_reward()

    # create the arrays that respresent the faces of the cube for visualization purposes.

    def get_state(self):
        return np.concatenate([face.ravel() for face in self.render()])

    def get_reward(self):
        reward_cube = np.ones((self.size, self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    block = self.cube[i,j,k]
                    for color in block.colors:
                        if color.orientation != color.color:
                            reward_cube[i,j,k] = 0
                            break

        reward = np.sum(reward_cube)
        if reward == self.size ** 3:
            return 100
        return reward

    def render(self):

        def get_colour(block, face):
            for color in block.colors:
                if color.orientation == face:
                    return color.color

        color_mapper = np.vectorize(get_colour)
        top = color_mapper(self.cube[0, :, :], 0)
        bottom = color_mapper(self.cube[self.size-1, :, :], 5)
        right = color_mapper(self.cube[:, self.size - 1, :], 2)
        left = color_mapper(self.cube[:, 0, :], 3)
        front = color_mapper(self.cube[:, :, self.size - 1], 4)
        back = color_mapper(self.cube[:, :, 0], 1)

        # perform rotations for visualization purposes
        return [np.rot90(top), np.rot90(back, k=2), np.rot90(right, k=2), left, front, np.rot90(bottom, k=3)]



    # Initialize a cube in a state in which all blocks are in the right place.
    def initialize_clean_cube(self):
        cube = np.empty((self.size, self.size, self.size), dtype=object)
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    # Core pieces
                    if i > 0 and i < self.size - 1 and j > 0 and j < self.size - 1 and k > 0 and k < self.size - 1:
                        cube[i,j,k] = CorePiece()

                    # Center Pieces

                    # BOTTOM
                    elif i == 0 and j > 0 and j < self.size - 1 and k > 0 and k < self.size - 1:
                        cube[i,j,k] = CenterPiece(ColorOrientation(Color.WHITE, Orientation.TOP))

                    # TOP
                    elif i == self.size - 1 and j > 0 and j < self.size - 1 and k > 0 and k < self.size - 1:
                        cube[i,j,k] = CenterPiece(ColorOrientation(Color.YELLOW, Orientation.BOTTOM))

                    # LEFT
                    elif i > 0 and i < self.size - 1 and j == 0 and k > 0 and k < self.size - 1:
                        cube[i, j, k] = CenterPiece(ColorOrientation(Color.ORANGE, Orientation.LEFT))

                    # RIGHT
                    elif i > 0 and i < self.size - 1 and j == self.size - 1 and k > 0 and k < self.size - 1:
                        cube[i, j, k] = CenterPiece(ColorOrientation(Color.RED, Orientation.RIGHT))

                    # BACK
                    elif i > 0 and i < self.size - 1 and j > 0 and j < self.size - 1 and k == 0:
                        cube[i, j, k] = CenterPiece(ColorOrientation(Color.BLUE, Orientation.BACK))

                    # FRONT
                    elif i > 0 and i < self.size - 1 and j > 0 and j < self.size - 1 and k == self.size - 1:
                        cube[i, j, k] = CenterPiece(ColorOrientation(Color.GREEN, Orientation.FRONT))

                    # Edge Pieces

                    # BOTTOM-LEFT
                    elif i == 0 and j == 0 and k > 0 and k < self.size - 1:
                        cube[i,j,k] = EdgePiece(ColorOrientation(Color.ORANGE, Orientation.LEFT), ColorOrientation(Color.WHITE, Orientation.TOP))

                    # BOTTOM-RIGHT
                    elif i == 0 and j == self.size - 1 and k > 0 and k < self.size - 1:
                        cube[i,j,k] = EdgePiece(ColorOrientation(Color.RED, Orientation.RIGHT), ColorOrientation(Color.WHITE, Orientation.TOP))

                    # BOTTOM-FRONT
                    elif i == 0 and j > 0 and j < self.size - 1 and k == self.size - 1:
                        cube[i, j, k] = EdgePiece(ColorOrientation(Color.GREEN, Orientation.FRONT),
                                                  ColorOrientation(Color.WHITE, Orientation.TOP))
                    # BOTTOM-BACK
                    elif i == 0 and j > 0 and j < self.size - 1 and k == 0:
                        cube[i, j, k] = EdgePiece(ColorOrientation(Color.BLUE, Orientation.BACK),
                                                  ColorOrientation(Color.WHITE, Orientation.TOP))
                    # FRONT-LEFT
                    elif i > 0 and i < self.size - 1 and j == 0 and k == self.size - 1:
                        cube[i,j,k] = EdgePiece(ColorOrientation(Color.ORANGE, Orientation.LEFT), ColorOrientation(Color.GREEN, Orientation.FRONT))

                    # BACK-LEFT
                    elif i > 0 and i < self.size - 1 and j == 0 and k == 0:
                        cube[i, j, k] = EdgePiece(ColorOrientation(Color.ORANGE, Orientation.LEFT),
                                                  ColorOrientation(Color.BLUE, Orientation.BACK))
                    # FRONT-RIGHT
                    elif i > 0 and i < self.size - 1 and j == self.size - 1 and k == self.size - 1:
                        cube[i, j, k] = EdgePiece(ColorOrientation(Color.RED, Orientation.RIGHT),
                                                  ColorOrientation(Color.GREEN, Orientation.FRONT))
                    # BACK-RIGHT
                    elif i > 0 and i < self.size - 1 and j == self.size - 1 and k == 0:
                        cube[i, j, k] = EdgePiece(ColorOrientation(Color.RED, Orientation.RIGHT),
                                                  ColorOrientation(Color.BLUE, Orientation.BACK))
                    # TOP-LEFT
                    elif i == self.size - 1 and j == 0 and k > 0 and k < self.size - 1:
                        cube[i, j, k] = EdgePiece(ColorOrientation(Color.ORANGE, Orientation.LEFT),
                                                  ColorOrientation(Color.YELLOW, Orientation.BOTTOM))
                    # TOP-RIGHT
                    elif i == self.size - 1 and j == self.size - 1 and k > 0 and k < self.size - 1:
                        cube[i, j, k] = EdgePiece(ColorOrientation(Color.RED, Orientation.RIGHT),
                                                  ColorOrientation(Color.YELLOW, Orientation.BOTTOM))
                    # TOP-FRONT
                    elif i == self.size - 1 and j > 0 and j < self.size - 1 and k == self.size - 1:
                        cube[i, j, k] = EdgePiece(ColorOrientation(Color.GREEN, Orientation.FRONT), ColorOrientation(Color.YELLOW, Orientation.BOTTOM))

                    # TOP-BACK
                    elif i == self.size - 1 and j > 0 and j < self.size - 1 and k == 0:
                        cube[i, j, k] = EdgePiece(ColorOrientation(Color.BLUE, Orientation.BACK),
                                                  ColorOrientation(Color.YELLOW, Orientation.BOTTOM))

                    # Corner pieces

                    # FRONT-TOP-LEFT
                    elif i == self.size - 1 and j == 0 and k == self.size - 1:
                        cube[i, j, k] = CornerPiece(ColorOrientation(Color.GREEN, Orientation.FRONT),
                                                    ColorOrientation(Color.ORANGE, Orientation.LEFT),
                                                    ColorOrientation(Color.YELLOW, Orientation.BOTTOM)
                                                    )
                    # FRONT-TOP-RIGHT
                    elif i == self.size - 1 and j == self.size - 1 and k == self.size - 1:
                        cube[i, j, k] = CornerPiece(ColorOrientation(Color.GREEN, Orientation.FRONT),
                                                    ColorOrientation(Color.RED, Orientation.RIGHT),
                                                    ColorOrientation(Color.YELLOW, Orientation.BOTTOM)
                                                    )
                    # FRONT-BOTTOM-RIGHT
                    elif i == 0 and j == self.size - 1 and k == self.size - 1:
                        cube[i, j, k] = CornerPiece(ColorOrientation(Color.GREEN, Orientation.FRONT),
                                                ColorOrientation(Color.RED, Orientation.RIGHT),
                                                ColorOrientation(Color.WHITE, Orientation.TOP)
                                                )
                    # FRONT-BOTTOM-LEFT
                    elif i == 0 and j == 0 and k == self.size - 1:
                        cube[i, j, k] = CornerPiece(ColorOrientation(Color.GREEN, Orientation.FRONT),
                                                ColorOrientation(Color.ORANGE, Orientation.LEFT),
                                                ColorOrientation(Color.WHITE, Orientation.TOP)
                                                )

                    # BACK-TOP-LEFT
                    elif i == self.size - 1 and j == 0 and k == 0:
                        cube[i, j, k] = CornerPiece(ColorOrientation(Color.BLUE, Orientation.BACK),
                                                    ColorOrientation(Color.ORANGE, Orientation.LEFT),
                                                    ColorOrientation(Color.YELLOW, Orientation.BOTTOM)
                                                    )
                    # BACK-TOP-RIGHT
                    elif i == self.size - 1 and j == self.size - 1 and k == 0:
                        cube[i, j, k] = CornerPiece(ColorOrientation(Color.BLUE, Orientation.BACK),
                                                    ColorOrientation(Color.RED, Orientation.RIGHT),
                                                    ColorOrientation(Color.YELLOW, Orientation.BOTTOM)
                                                    )
                    # BACK-BOTTOM-RIGHT
                    elif i == 0 and j == self.size - 1 and k == 0:
                        cube[i, j, k] = CornerPiece(ColorOrientation(Color.BLUE, Orientation.BACK),
                                                    ColorOrientation(Color.RED, Orientation.RIGHT),
                                                    ColorOrientation(Color.WHITE, Orientation.TOP)
                                                    )
                    # BACK-BOTTOM-LEFT
                    elif i == 0 and j == 0 and k == 0:
                        cube[i, j, k] = CornerPiece(ColorOrientation(Color.BLUE, Orientation.BACK),
                                                    ColorOrientation(Color.ORANGE, Orientation.LEFT),
                                                    ColorOrientation(Color.WHITE, Orientation.TOP)
                                                    )
        return cube
class Axis:
    X = 0
    Y = 1
    Z = 2

class Orientation:
    TOP=0
    BACK=1
    RIGHT=2
    LEFT=3
    FRONT=4
    BOTTOM=5

class Color:
    WHITE=0
    BLUE=1
    RED=2
    ORANGE=3
    GREEN=4
    YELLOW=5

class ColorOrientation:
    def __init__(self, color, orientation):
        self.color = color
        self.orientation = orientation

class Block:
    def __init__(self):
        self.colors = []

    # TOP and BOTTOM remain constant
    def rotate_x(self):
        for color in self.colors:
            match color.orientation:
                case Orientation.LEFT:
                    color.orientation = Orientation.BACK
                case Orientation.BACK:
                    color.orientation = Orientation.RIGHT
                case Orientation.RIGHT:
                    color.orientation = Orientation.FRONT
                case Orientation.FRONT:
                    color.orientation = Orientation.LEFT

    # RIGHT and LEFT remain constant
    def rotate_y(self):
        for color in self.colors:
            match color.orientation:
                case Orientation.FRONT:
                    color.orientation = Orientation.TOP
                case Orientation.TOP:
                    color.orientation = Orientation.BACK
                case Orientation.BACK:
                    color.orientation = Orientation.BOTTOM
                case Orientation.BOTTOM:
                    color.orientation = Orientation.FRONT

    # Front and Back remain constant
    def rotate_z(self):
        for color in self.colors:
            match color.orientation:
                case Orientation.BOTTOM:
                    color.orientation = Orientation.RIGHT
                case Orientation.LEFT:
                    color.orientation = Orientation.BOTTOM
                case Orientation.TOP:
                    color.orientation = Orientation.LEFT
                case Orientation.RIGHT:
                    color.orientation = Orientation.TOP

class CorePiece(Block):
    def __init__(self):
        super().__init__()
class CenterPiece(Block):
    def __init__(self, color):
        super().__init__()
        self.colors.append(color)

class EdgePiece(Block):
    def __init__(self, first_color, second_color):
        super().__init__()
        self.colors.append(first_color)
        self.colors.append(second_color)

class CornerPiece(Block):
    def __init__(self, first_color, second_color, third_color):
        super().__init__()
        self.colors.append(first_color)
        self.colors.append(second_color)
        self.colors.append(third_color)

