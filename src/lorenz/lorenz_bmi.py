from bmipy import Bmi
import numpy as np
from lorenz import utils
from typing import Any, Tuple
from dateutil.parser import parse
import pandas as pd


def rk4(state, dt, F):
    k1 = f(state, F)
    k2 = f(state + 0.5 * dt * k1, F)
    k3 = f(state + 0.5 * dt * k2, F)
    k4 = f(state + dt * k3, F)
    return (1 / 6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)


def f(state, F):
    J = len(state)
    k = np.zeros(J)

    k[0] = (state[1] - state[J - 2]) * state[J - 1] - state[0]
    k[1] = (state[2] - state[J - 1]) * state[0] - state[1]
    k[J - 1] = (state[0] - state[J - 3]) * state[J - 2] - state[J - 1]

    for j in range(2, J - 1):
        k[j] = (state[j + 1] - state[j - 2]) * state[j - 1] - state[j]

    return k + F


class Lorenz(Bmi):
    """Lorenz model wrapped in a BMI interface."""

    _var_units = {'state': '[-]',
                  "latitude": 'm',
                  "longitude":"m",
                   }
    _name = 'Example Python Lorenz model, BMI'
    _input_var_names = ['state']
    _output_var_names = ['state']

    def __init__(self):
        self._dt = 0
        self._t = 0.
        self._startTime = 0.
        self._endTime = 0.
        self.start_time = np.datetime64("1970-01-01T00:00:00")
        self.end_time = np.datetime64("1970-01-01T00:00:00")

        self._state = None

        self._value = {}

        self._shape = (0, 0)
        self._spacing = (0., 0.)
        self._origin = (0., 0.)

        self._J = 0
        self._F = 0

        self._settings = {}

    def initialize(self, config_file):

        settings: dict[str, Any] = utils.read_config(config_file)
        self._settings = settings

        self._dt = settings['dt']
        self._t = 0.

        time_delta = parse(settings['end_time']) - parse(settings['start_time'])
        # whole days
        self._startTime = 0
        self._endTime = time_delta.days + time_delta.seconds / (3600 * 24)

        # equivalent np.dt64
        self.start_time = pd.Timestamp(parse(settings['start_time'])).to_datetime64()
        self.end_time = pd.Timestamp(parse(settings['end_time'])).to_datetime64()

        self._J = settings['J']
        self._F = settings['F']

        self._state = np.array(settings['start_state'])

        self._value['state'] = "_state"

        self._shape = (self._J, 1)
        self._spacing = (1., 1.)
        self._origin = (0., 0.)

    def update(self):
        if self._t >= self._endTime:
            raise "endTime already reached, model not updated"
        self._state = self._state + rk4(self._state, self._dt, self._F)

        self._t += self._dt

    def update_until(self, t):
        if (t < self._t) or t > self._endTime:
            raise "wrong time input: smaller than model time or larger than endTime"
        while self._t < t:
            self.update()

    def finalize(self):
        self._dt = 0
        self._t = 0

        self._state = None

    def get_var_type(self, var_name):
        return str( getattr(self, self.get_value_ptr(var_name)).dtype)

    def get_var_units(self, var_name):
        return self._var_units[var_name]

    def get_var_rank(self, var_name):
        return self.get_value(var_name, np.zeros(self._J)).ndim

    def get_value(self, var_name, dest):
        dest[:] = getattr(self, self._value[var_name])
        return dest

    def get_value_at_indices(self, var_name: str, dest: np.ndarray, indices: int) -> np.ndarray:
        dest[:] = getattr(self, self._value[var_name])[indices]
        return dest

    def set_value(self, var_name, src):
        val = getattr(self, self._value[var_name])
        val[:] = src

    def set_value_at_indices(self, var_name, indices, src):
        val = getattr(self, self._value[var_name])
        val[indices] = src

    def get_component_name(self):
        return self._name

    def get_input_var_names(self):
        return self._input_var_names

    def get_output_var_names(self):
        return self._output_var_names

    def get_grid_shape(self, grid_id, shape):
        """Number of rows and columns of uniform rectilinear grid."""
        return self._shape

    def get_grid_spacing(self, grid_id, spacing):
        spacing[:] = self._spacing
        return spacing

    def get_grid_origin(self, grid_id, origin):
        """Origin of uniform rectilinear grid."""
        origin[:] = self._origin
        return origin

    def get_grid_type(self, var_name):
        if var_name in self._value:
            return "rectilinear"
        else:
            return None

    def get_var_itemsize(self, var_name: str) -> int:
        return getattr(self, self.get_value_ptr(var_name)).itemsize

    def get_var_nbytes(self, var_name: str) -> int:
        return getattr(self, self.get_value_ptr(var_name)).nbytes

    # Grid information
    def get_var_grid(self, name: str) -> int:
        return 0

    def get_grid_rank(self, grid: int) -> int:
        return len(self._shape)

    def get_grid_size(self, grid: int) -> int:
        return int(np.prod(self._shape))

    def get_start_time(self):
        return get_unixtime(self.start_time)

    def get_end_time(self):
        return get_unixtime(self.end_time)

    def get_current_time(self):
        """"Weird switching but should now return in seconds since 1970"""
        current = self.start_time + np.timedelta64(int(self._t * 24 * 3600 * 1000), "ms")
        return get_unixtime(current)

    def get_time_units(self):
        return "seconds since 1970-01-01 00:00:00.0 +0000"

    def get_time_step(self):
        """Model is in days so return seconds dt"""
        return self._dt * 24 * 3600

    # not implemented & not planning to
    def get_grid_x(self, grid: int, x: np.ndarray) -> np.ndarray:
        x[:] = np.arange(self._origin[1], self._shape[1], self._spacing[1])
        return x

    def get_grid_y(self, grid: int, y: np.ndarray) -> np.ndarray:
        y[:] = np.arange(self._origin[0], self._shape[0], self._spacing[0])
        return y

    def get_input_item_count(self) -> int:
        raise NotImplementedError()

    def get_output_item_count(self) -> int:
        raise NotImplementedError()

    def get_var_location(self, name: str) -> str:
        raise NotImplementedError()

    def get_grid_z(self, grid: int, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def get_grid_node_count(self, grid: int) -> int:
        raise NotImplementedError()

    def get_grid_edge_count(self, grid: int) -> int:
        raise NotImplementedError()

    def get_grid_face_count(self, grid: int) -> int:
        raise NotImplementedError()

    def get_grid_edge_nodes(self, grid: int, edge_nodes: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def get_grid_face_edges(self, grid: int, face_edges: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def get_grid_face_nodes(self, grid: int, face_nodes: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def get_grid_nodes_per_face(
            self, grid: int, nodes_per_face: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def get_value_ptr(self, var_name):
        """Reference to values.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        array_like
            Value array.
        """
        return self._value[var_name]


def get_unixtime(dt64: np.datetime64) -> float:
    """Get unix timestamp (seconds since 1 january 1970) from a np.datetime64."""
    return dt64.astype("datetime64[s]").astype("int")
