from .solvers import FixedGridODESolver
from .rk_common import rk4_alt_step_func, rk3_step_func
from .misc import Perturb

class Euler(FixedGridODESolver):
    order = 1

    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        return dt * f0, f0


class Midpoint(FixedGridODESolver):
    order = 2

    def _step_func(self, func, t0, dt, t1, y0):
        half_dt = 0.5 * dt
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        y_mid = y0 + f0 * half_dt
        y_mid = reflect(y_mid)
        return dt * func(t0 + half_dt, y_mid), f0


class RK4(FixedGridODESolver):
    order = 4

    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        return rk4_alt_step_func(func, t0, dt, t1, y0, f0=f0, perturb=self.perturb), f0


class Heun3(FixedGridODESolver):
    order = 3

    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)

        butcher_tableu = [
            [0.0, 0.0, 0.0, 0.0],
            [1/3, 1/3, 0.0, 0.0],
            [2/3, 0.0, 2/3, 0.0],
            [0.0, 1/4, 0.0, 3/4],
        ]

        return rk3_step_func(func, t0, dt, t1, y0, reflect_fn=self.reflect_fn, reflect_velocity_fn=self.reflect_velocity_fn, butcher_tableu=butcher_tableu, f0=f0, perturb=self.perturb), f0
