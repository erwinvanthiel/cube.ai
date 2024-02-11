import random

from src.environment import Environment
import numpy as np
import math
# This class represents the Rubik's Cube environment.
class CubeEnvironment(Environment):
    def __init__(self, size):
        super(CubeEnvironment, self).__init__()
        self.size = size
        self.faces = self.initialize_clean_cube()
        self.state = self.get_flat_state()

    # This method represents the action of turning a slice of the cube.
    # It returns the reward and an indication of whether the terminal state has been reached
    def turn_slice(self, axis, index, clock_wise=True):
        match axis:
            case Axis.X:
                affected_faces = [Orientation.FRONT, Orientation.RIGHT, Orientation.BACK, Orientation.LEFT]
            case Axis.Y:
                affected_faces = [Orientation.FRONT, Orientation.TOP, Orientation.BACK, Orientation.BOTTOM]
            case Axis.Z:
                affected_faces = [Orientation.TOP, Orientation.RIGHT, Orientation.BOTTOM, Orientation.LEFT]
            case _:
                raise Exception("Unrecognized plane {0}".format(axis))

        reps = 1
        if not clock_wise:
            reps = 3
        for i in range(reps):

            # rotate middle pieces
            if axis == Axis.X:
                temp = self.faces[affected_faces[0], index, :].copy()
                self.faces[affected_faces[0], index, :] = self.faces[affected_faces[1], index, :]
                self.faces[affected_faces[1], index, :] = self.faces[affected_faces[2], index, :]
                self.faces[affected_faces[2], index, :] = self.faces[affected_faces[3], index, :]
                self.faces[affected_faces[3], index, :] = temp
                # rotate middle faces
                if index == 0:
                    self.faces[Orientation.BOTTOM] = np.rot90(self.faces[Orientation.BOTTOM])
                elif index == self.size - 1:
                    self.faces[Orientation.TOP] = np.rot90(self.faces[Orientation.TOP])

            if axis == Axis.Y:
                temp = self.faces[affected_faces[0], :, index].copy()
                self.faces[affected_faces[0], :, index] = self.faces[affected_faces[1], :, index]
                self.faces[affected_faces[1], :, index] = self.faces[affected_faces[2], :, index]
                self.faces[affected_faces[2], :, index] = self.faces[affected_faces[3], :, index]
                self.faces[affected_faces[3], :, index] = temp

                if index == 0:
                    self.faces[Orientation.LEFT] = np.rot90(self.faces[Orientation.LEFT])
                elif index == self.size - 1:
                    self.faces[Orientation.RIGHT] = np.rot90(self.faces[Orientation.RIGHT])

            if axis == Axis.Z:
                temp = self.faces[affected_faces[0], index, :].copy()
                self.faces[affected_faces[0], index, :] = self.faces[affected_faces[1], :, index]
                self.faces[affected_faces[1], :, index] = self.faces[affected_faces[2], index, :]
                self.faces[affected_faces[2], index, :] = self.faces[affected_faces[3], :, index]
                self.faces[affected_faces[3], :, index] = temp

                if index == 0:
                    self.faces[Orientation.BACK] = np.rot90(self.faces[Orientation.BACK])
                elif index == self.size - 1:
                    self.faces[Orientation.FRONT] = np.rot90(self.faces[Orientation.FRONT])

        return self.get_reward()

    def random_scramble(self, k):
        self.faces = self.initialize_clean_cube()
        for i in range(random.Random().randint(1, k)):
            self.perform_action(random.Random().randint(0, (3 * self.size) - 1))

    # action_slice depicts what slice gets turned, e.g. 2 is axis 0 a.k.a X and then the third (index = 2) slice.
    # If we put in action=4 we action_slice will be 2, so we will turn the third slice around the x-dimension.
    # aciton=5 will do the same but then counter-clockwise
    def perform_action(self, action):
        clock_wise = action % 2 == 0
        action_slice = math.floor(action / 2)
        return self.turn_slice(math.floor(action_slice / 3), action_slice % self.size, clock_wise=clock_wise)

    def get_flat_state(self):
        return np.concatenate([face.ravel() for face in self.faces])

    def get_reward(self):
        top_complete = (self.faces[Orientation.TOP][self.faces[Orientation.TOP] == Orientation.TOP] + 1).sum() == self.size ** 2
        bottom_complete = (self.faces[Orientation.BOTTOM][self.faces[Orientation.BOTTOM] == Orientation.BOTTOM] / 5).sum() == self.size ** 2
        number_of_layers = 0
        for i in range(self.size):
            for j in range(6):
                if len(self.faces[j, i, :][self.faces[j, i, :] == j]) != self.size:
                    return number_of_layers * int(top_complete), False
            number_of_layers += 1
        if bottom_complete:
            return number_of_layers * int(top_complete), True
        return (number_of_layers - 1) * int(top_complete), False

    # create the arrays that represent the faces of the cube for visualization purposes.
    def get_faces(self):
        return self.faces

    # Initialize a cube in a state in which all blocks are in the right place.
    def initialize_clean_cube(self):
        faces = np.ones((6, self.size, self.size), dtype=int)
        for i in range(6):
            faces[i, :, :] *= i
        return faces

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