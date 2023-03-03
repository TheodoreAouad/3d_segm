from typing import Callable, Tuple
import time
from IPython.display import clear_output

import numpy as np
import matplotlib.pyplot as plt


class PointDichotomy:
    def __init__(
        self, W: np.ndarray, dist_fn: Callable, w_values: np.ndarray = None, N: int = None,
    ):
        self.W = W
        self.dist_fn = dist_fn
        self.w_values = w_values if w_values is not None else np.unique(W)
        self.N = N if N is not None else len(self.w_values)

        self.index = None
        self.S, self.dist_S = None, None
        self.Sb, self.dist_Sb = None, None
        self.Sa, self.dist_Sa = None, None

        self.derivative = None

        self.index_history = []

    @property
    def objective_value(self):
        return self.dist_S

    @property
    def w_value(self):
        return self.w_values[self.index]

    def update_history(self):
        self.index_history.append(self.index)

    def update_state(self, new_idx: int):
        self.index = new_idx

        self.S = self.W >= self.w_values[self.index]
        self.dist_S = self.dist_fn(self.W, self.S)

        if new_idx != 0:
            self.Sb = self.W >= self.w_values[self.index - 1]
            self.dist_Sb = self.dist_fn(self.W, self.Sb)

        if new_idx != self.N - 1:
            self.Sa = self.W >= self.w_values[self.index + 1]
            self.dist_Sa = self.dist_fn(self.W, self.Sa)

        self.derivative = self.get_state_derivative(self.dist_S, self.dist_Sb, self.dist_Sa)
        return self

    @staticmethod
    def get_state_derivative(dist_S, dist_Sb=None, dist_Sa=None):
        if dist_Sb is None:
            return (np.sign(dist_Sa - dist_S), np.sign(dist_Sa - dist_S))

        if dist_Sa is None:
            return (np.sign(dist_S - dist_Sb), np.sign(dist_S - dist_Sb))

        return (np.sign(dist_S - dist_Sb), np.sign(dist_Sa - dist_S))

    def __repr__(self):
        return f"PointDichotomy(index={self.index}, dist_S={self.dist_S:.2e})"

    def set_state(self, new_pt: "PointDichotomy"):
        self.index = new_pt.index
        self.S, self.dist_S = new_pt.S, new_pt.dist_S
        self.Sb, self.dist_Sb = new_pt.Sb, new_pt.dist_Sb
        self.Sa, self.dist_Sa = new_pt.Sa, new_pt.dist_Sa
        self.derivative = new_pt.derivative

        return self


class DichotomyBinarization:
    LOCAL_MAX = (1, -1)
    LOCAL_MIN = (-1, 1)
    INCREASING = (1, 1)
    DECREASING = (-1, -1)

    def __init__(self, W: np.ndarray, dist_fn: Callable, init: Tuple[int, int] = None):
        self.W = W
        self.dist_fn = dist_fn
        self.init = init
        self.w_values = np.unique(W)
        self.N = len(self.w_values)

        if self.init is None:
            self.init = (0, self.N - 1)

        # self.min_index = 0
        # self.max_index = self.N - 1

        self.pt1 = self.new_point()
        self.pt2 = self.new_point()
        self.pt_tmp = self.new_point()

    def new_point(self):
        return PointDichotomy(W=self.W, dist_fn=self.dist_fn, w_values=self.w_values, N=self.N)


    def find_best_point(self):
        best_pts = [self.new_point().set_state(self.find_best_point_bounds(*self.init))]
        best_pts.append(self.new_point().set_state(self.find_best_point_bounds(self.init[0], best_pts[0].index - 1)))
        best_pts.append(self.new_point().set_state(self.find_best_point_bounds(best_pts[0].index + 1, self.init[1])))


        best_pt = best_pts[np.argmin([pt.objective_value for pt in best_pts])]
        return best_pt


    def find_best_point_bounds(self, min_index: int = None, max_index: int = None):
        if min_index is None:
            min_index = self.init[0]
        if max_index is None:
            max_index = self.init[1]

        self.pt1.update_state(min_index)
        self.pt2.update_state(max_index)

        self.pt1.update_history()
        self.pt2.update_history()

        if self.pt1.derivative == self.LOCAL_MAX:
            self.best_pt = self.pt1
            return self.pt1

        if self.pt2.derivative == self.LOCAL_MAX:
            self.best_pt = self.pt2
            return self.pt2

        if self.pt1.derivative == self.DECREASING and self.pt2.derivative == self.DECREASING:
            self.best_pt = self.pt1
            return self.pt1

        if self.pt1.derivative == self.INCREASING and self.pt2.derivative == self.INCREASING:
            self.best_pt = self.pt2
            return self.pt2

        for _ in range(self.N):
            pt1_der = self.pt1.derivative
            pt2_der = self.pt2.derivative

            if pt1_der == self.LOCAL_MIN:
                pt1_der = self.INCREASING if np.random.rand() > .5 else self.DECREASING
            if pt2_der == self.LOCAL_MIN:
                pt2_der = self.INCREASING if np.random.rand() > .5 else self.DECREASING

            # if pt1_der == self.INCREASING and pt2_der == self.DECREASING:
            if pt1_der != pt2_der:
                self.pt_tmp.update_state((self.pt1.index + self.pt2.index) // 2)

                if self.pt_tmp.derivative == self.LOCAL_MAX:
                    self.pt_tmp.update_history()
                    self.best_pt = self.pt_tmp
                    return self.pt_tmp

                if self.pt_tmp.derivative == self.INCREASING:
                    self.pt1.set_state(self.pt_tmp)

                elif self.pt_tmp.derivative == self.DECREASING:
                    self.pt2.set_state(self.pt_tmp)

            elif pt1_der == self.INCREASING and pt2_der == self.INCREASING:
                self.pt_tmp.update_state((min_index + self.pt1.index) // 2)

                if self.pt_tmp.derivative == self.LOCAL_MAX:
                    self.pt_tmp.update_history()
                    self.best_pt = self.pt_tmp
                    return self.pt_tmp

                self.pt1.set_state(self.pt_tmp)

            elif pt1_der == self.DECREASING and pt2_der == self.DECREASING:
                self.pt_tmp.update_state((max_index + self.pt2.index) // 2)

                if self.pt_tmp.derivative == self.LOCAL_MAX:
                    self.pt_tmp.update_history()
                    self.best_pt = self.pt_tmp
                    return self.pt_tmp

                self.pt2.set_state(self.pt_tmp)

            for pt in [self.pt1, self.pt2, self.pt_tmp]:
                pt.update_history()


        return self.pt_tmp

    @property
    def n_steps(self):
        return len(self.pt1.index_history)

    def animate(self, wait=None,):
        self.all_pt1, self.all_pt2, self.cur_pt_tmp = [], [], PointDichotomy(W=self.W, dist_fn=self.dist_fn, w_values=self.w_values)
        for idx in range(len(self.pt1.index_history)):
            clear_output(wait=True)
            # if ax is None:
            fig, ax = plt.subplots()

            self._animate_step(idx, ax=ax)
            ax.set_title(f"Step {idx}")
            ax.legend()

            plt.show()
            if wait is not None:
                time.sleep(wait)

    def _animate_step(self, idx, ax=None):
        self.all_pt1.append(PointDichotomy(W=self.W, dist_fn=self.dist_fn, w_values=self.w_values).update_state(self.pt1.index_history[idx]))
        self.all_pt2.append(PointDichotomy(W=self.W, dist_fn=self.dist_fn, w_values=self.w_values).update_state(self.pt2.index_history[idx]))

        if len(self.pt_tmp.index_history) > 0:
            self.cur_pt_tmp.update_state(self.pt_tmp.index_history[idx])

        self.plot_points(ax=ax)

    def plot_points(self, ax):
        # Previous points
        ax.scatter([pt1.w_value for pt1 in self.all_pt1[:-1]], [pt1.objective_value for pt1 in self.all_pt1[:-1]], marker="o")
        ax.scatter([pt2.w_value for pt2 in self.all_pt2[:-1]], [pt2.objective_value for pt2 in self.all_pt2[:-1]], marker="o")
        # Current points
        ax.scatter([self.all_pt1[-1].w_value], [self.all_pt1[-1].objective_value], c="r", label="pt1", marker="s")
        ax.scatter([self.all_pt2[-1].w_value], [self.all_pt2[-1].objective_value], c="magenta", label="pt2", marker="s")

        # Temporary point
        if len(self.pt_tmp.index_history) > 0:
            ax.scatter([self.cur_pt_tmp.w_value], [self.cur_pt_tmp.objective_value], c="g", label="tmp")
        ax.grid()

    def animate_on_curve(self, values, objectives, best_value=None, wait=None, figsize=(10, 5)):
        self.all_pt1, self.all_pt2, self.cur_pt_tmp = [], [], PointDichotomy(W=self.W, dist_fn=self.dist_fn, w_values=self.w_values)
        for idx in range(len(self.pt1.index_history)):
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=figsize)

            ax.plot(values, objectives, label="objective",)
            if best_value is not None:
                ax.axvline(best_value, c="k", label=r"$S^*$", linestyle="--", alpha=.5, marker="o")
            # ax.plot([best_pt.w_value], [best_pt.objective_value], c="g", marker="o", label="best_pt")

            self._animate_step(idx, ax=ax)
            ax.set_title(f"Best: {best_value:.2e}, Step {idx}")

            ax.legend()

            plt.show()
            if wait is not None:
                time.sleep(wait)
