# Credits: https://github.com/inikishev/image-descent/blob/main/image_descent/image_descent.py, initial code by @inikishev, updated for velocity plots of optimization path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib import cm
from typing import Optional, Any, Callable, Sequence, Iterable
import numpy as np
import torch

def flatten(iterable: Iterable) -> list[Any]:
    if isinstance(iterable, Iterable):
        return [a for i in iterable for a in flatten(i)]
    else:
        return [iterable]

class Compose:
    def __init__(self, *args):
        self.transforms = flatten(args)
    def __call__(self, x, *args, **kwargs):
        for t in self.transforms:
            if t is not None:
                x = t(x, *args, **kwargs)
        return x

def ax_plot(ax: Axes, *data, title=None, ylim=None, xlabel=None, ylabel=None):
    ax.plot(*data)
    if title:
        ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_major_locator(AutoLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator('auto'))  # type:ignore
    ax.yaxis.set_minor_locator(AutoMinorLocator('auto'))  # type:ignore
    ax.grid(which="major", color='black', alpha=0.09)
    ax.grid(which="minor", color='black', alpha=0.04)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax

class FunctionDescent2D(torch.nn.Module):
    def __init__(
        self,
        fn: Callable | Sequence[Callable],
        coords: Optional[Sequence[int | float] | Callable] = None,
        dtype: torch.dtype = torch.float32,
        before_step: Optional[Callable | Sequence[Callable]] = None,
        after_step: Optional[Callable | Sequence[Callable]] = None,
        normalize: Optional[int] = 100,
        mode: str = 'unpacked',
        xlim: Optional[Sequence[int|float]] = None,
        ylim: Optional[Sequence[int|float]] = None,
        lims_from_surface: bool = True,
        minimum: Optional[Sequence[int|float]]  = None
    ):
        """Perform gradient descent on a 2D function

        Args:
            fn (Callable | Sequence[Callable]):
            The function to optimize, must accept two pytorch scalars and return a pytorch tensor, e.g.
            ```py
            def rosenbrock(x:torch.Tensor, y:torch.Tensor):
                return (1-x)**2 + 100*(y-x**2)**2
            ```

            coords (Optional[Sequence[int  |  float]  |  Callable], optional):
            Initial coordinates, e.g. `(x, y)`. If `fn` has `start` method and this is None, it will be used to get the coordinates.
            Otherwise this must be specified. Defaults to None.

            dtype (torch.dtype, optional):
            Data type in which calculations will be performed. Defaults to torch.float32.

            before_step (Optional[Callable  | Sequence[Callable]], optional):
            Optional function or sequence of functions that gets applied to coordinates before each step, and is part of the backpropagation.
            Can be used to add noise, etc. Defaults to None. Example:
            ```py
            def add_noise(coords:torch.Tensor):
                return coords + torch.randn_like(coords) * 0.01
            ```

            after_step (Optional[Callable | Sequence[Callable]], optional):
            Optional function or sequence of functions that gets applied to loss after each step, and is part of the backpropagation.
            Defaults to None. Example:
            ```py
            def pow_loss(loss:torch.Tensor):
                return loss**2
            ```

            normalize (int, optional):
            If not None, adds normalization to 0-1 range to `fn` by calculating `normalize`*`normalize` grid of values and using minimum and maximum. Defaults to 100.

            mode (str, optional):
            `unpacked` means `fn` gets passed `coords[0], coords[1]`, `packed` means `fn` gets passed `coords` and currently doesn't work.
            Defaults to 'unpacked'.

            xlim (tuple[float, float], optional):
            Optionally specify x-axis limits for plotting as `(left, right)` tuple.
            Does not prevent optimizer going outside of the limit.
             If `fn` has `domain` method and this is None, it will be used to get the x and y limis.
            Defaults to None.

            ylim (tuple[float, float], optional):
            Optionally specify y-axis limits for plotting as `(top, bottom)` tuple.
            Does not prevent optimizer going outside of the limit.
            If `fn` has `domain` method and this is None, it will be used to get the x and y limis.
            Defaults to None.

            lims_from_surface (bool):
            Whether to get `xlim` and `ylim` from `fn` if it has `domain` method and `xlim` and `ylim` are `None`. Defaults to True.

            minimum (tuple[float, float], optional):
            Optional `(x,y)` coordinates of the global minimum.
            If specified, distance to minimum will be logged each step into `distance_to_minimum_history`.
            If `fn` has `minimum` method and this is None, it will be used to get the minimum.
            Defaults to None.

        Raises:
            ValueError: _description_
        """
        super().__init__()
        self.fn = Compose(fn)
        self.dtype = dtype
        self.mode = mode
        self.xlim = xlim
        self.ylim = ylim

        self.before_step = Compose(before_step)
        self.after_step = Compose(after_step)

        # get coords from surface if it has them
        if coords is None:
            if hasattr(fn, "start"): coords = fn.start() # type:ignore
            else: raise ValueError("No coords provided and surface has no coords method")

        # get coords
        if callable(coords): coords = coords()
        if isinstance(coords, np.ndarray): self.coords = torch.nn.Parameter(torch.from_numpy(coords).to(self.dtype))
        elif isinstance(coords, torch.Tensor): self.coords = torch.nn.Parameter(coords.to(self.dtype))
        else: self.coords = torch.nn.Parameter(torch.from_numpy(np.asanyarray(coords)).to(self.dtype))

        # get limits from surface if it has them
        if xlim is None and ylim is None and lims_from_surface:
            if hasattr(fn, "domain"): self.xlim, self.ylim = fn.domain() # type:ignore

        # get minimum from surface if it has it
        self.minimum = minimum
        if self.minimum is None:
            if hasattr(fn, "minimum"): self.minimum = fn.minimum() # type:ignore

        # history
        self.coords_history = []
        self.loss_history = []
        self.distance_to_minimum_history = []

        if normalize and (self.xlim is not None) and (self.ylim is not None):
            _, _, z = self.compute_image(steps=normalize)
            vmin, vmax = np.min(z), np.max(z)-np.min(z)
            self.fn = Compose(fn, lambda x, y: (x-vmin) / vmax)

    def forward(self):
        # have coords to history
        self.coords_history.append(self.coords.detach().cpu().clone()) # pylint:disable=E1102

        # save distance to minimum
        if self.minimum is not None: 
            self.distance_to_minimum_history.append(
                torch.norm(self.coords - torch.tensor(self.minimum, dtype=self.dtype)).detach().cpu().clone()
            )

        # get loss
        coords = self.before_step(self.coords)
        if self.mode == 'unpacked': 
            loss:torch.Tensor = self.fn(coords[0], coords[1])
        elif self.mode == 'packed': 
            loss:torch.Tensor = self.fn(coords)
        else: 
            raise ValueError(f"Unknown mode {self.mode}")
        loss = self.after_step(loss)

        # save loss to history
        self.loss_history.append(loss.detach().cpu().clone())

        return loss

    def step(self): 
        return self.forward()

    def compute_image(self, xlim=None, ylim=None, steps=1000, auto_expand=True):
        if xlim is None: xlim = self.xlim
        if ylim is None: ylim = self.ylim

        if (xlim is None or ylim is None) or auto_expand:
            if len(self.coords_history) == 0: 
                xvals, yvals = [-1,1],[-1,1]
            else: 
                xvals, yvals = list(zip(*self.coords_history))
            if xlim is None: xlim = min(xvals), max(xvals)
            if ylim is None: ylim = min(yvals), max(yvals)

        if auto_expand and len(self.coords_history) != 0:
            xlim = min(*xvals, xlim[0]), max(*xvals, xlim[1]) # type:ignore
            ylim = min(*yvals, ylim[0]), max(*yvals, ylim[1]) # type:ignore

        xstep = (xlim[1] - xlim[0]) / steps
        ystep = (ylim[1] - ylim[0]) / steps

        y, x = torch.meshgrid(
            torch.arange(ylim[0], ylim[1], ystep), 
            torch.arange(xlim[0], xlim[1], xstep), 
            indexing='xy'
        )
        z = [self.fn(xv, yv).numpy() for xv, yv in zip(x, y)]
        self.computed_image = (x.numpy(), y.numpy(), z)
        return self.computed_image

    def plot_image(self, xlim=None, ylim=None, cmap='gray', levels=20, figsize=None, show=False, return_fig=False):
        image = self.compute_image(xlim, ylim)
        fig, ax = plt.subplots(1, 1, figsize=figsize, layout='tight')
        ax.set_title("Loss Landscape")
        ax.set_frame_on(False)
        cmesh = ax.pcolormesh(*image, cmap=cmap, shading='auto', zorder=0)
        if levels:
            ax.contour(*image, linewidths=0.5, alpha=0.5, cmap='binary', levels=levels)
        current_coord = self.coords.detach().cpu() # pylint:disable=E1102
        minimum = self.minimum
        # ax.scatter([current_coord[0]], [current_coord[1]], s=4, color='red', label='Current Position')
        if minimum is not None: 
            ax.scatter(
                [minimum[0]], [minimum[1]], 
                s=64, c='lime', marker='+', zorder=4, alpha=0.7, label='Minimum'
            )
        fig.colorbar(cmesh, ax=ax, label='Loss')
        if show: plt.show()
        if return_fig: return fig, ax

    def plot_losses(self, figsize=None, show=False, return_fig=False):
        fig, ax = plt.subplots(1, 1, figsize=figsize, layout='tight')
        ax_plot(ax, range(len(self.loss_history)), self.loss_history, title="Loss History", xlabel="Step", ylabel="Loss")
        if show: plt.show()
        if return_fig: return fig, ax

    def plot_distance_to_minimum(self, figsize=None, show=False, return_fig=False):
        fig, ax = plt.subplots(1, 1, figsize=figsize, layout='tight')
        ax_plot(ax, range(len(self.distance_to_minimum_history)), self.distance_to_minimum_history, title="Distance to Minimum", xlabel="Step", ylabel="Distance")
        if show: plt.show()
        if return_fig: return fig, ax

    def plot_path(self, xlim=None, ylim=None, surface_cmap='gray', levels=20, figsize=None, show=False, return_fig=False):
        """
        Plots the optimization path on top of the loss landscape image.
        Color of the path represents the step size (blue=large steps, red=small steps).
        """
        # Compute the image and plot it
        fig, ax = self.plot_image(xlim=xlim, ylim=ylim, cmap=surface_cmap, levels=levels, figsize=figsize, return_fig=True)
        ax.set_title("Optimization Path with Step Sizes")
        ax.set_frame_on(False)

        if len(self.coords_history) < 2:
            print("Not enough steps to plot path with step sizes.")
            return fig, ax

        # Extract coordinates
        coords = np.array(self.coords_history)
        x = coords[:, 0]
        y = coords[:, 1]

        # Compute step sizes (Euclidean distance between consecutive points)
        step_sizes = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
        
        # Normalize step sizes for color mapping
        norm = Normalize(vmin=np.min(step_sizes), vmax=np.max(step_sizes))
        cmap = cm.viridis  # You can choose any other colormap

        # Create segments for LineCollection
        segments = [((x[i], y[i]), (x[i+1], y[i+1])) for i in range(len(x)-1)]

        # Create LineCollection
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=4)
        lc.set_array(step_sizes)
        lc.set_zorder(3)
        ax.add_collection(lc)

        # Add a colorbar for step sizes
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # cbar = fig.colorbar(sm, ax=ax, label='Step Size')

        # Plot markers for the first N-1 points
        # ax.scatter(x[:-1], y[:-1], c=step_sizes, cmap=cmap, norm=norm, s=16, zorder=4, label='Path Steps')

        # Optionally, plot the last point without a color (since it has no corresponding step size)
        # ax.scatter(x[-1], y[-1], c='black', s=16, zorder=5, label='Final Point')

        ax.legend()

        if show: 
            plt.show()
        if return_fig: 
            return fig, ax
            """
            Plots the optimization path on top of the loss landscape image.
            Color of the path represents the step size (blue=large steps, red=small steps).
            """
            # Compute the image and plot it
            fig, ax = self.plot_image(xlim=xlim, ylim=ylim, cmap=surface_cmap, levels=levels, figsize=figsize, return_fig=True)
            ax.set_title("Optimization Path with Step Sizes")
            ax.set_frame_on(False)

            if len(self.coords_history) < 2:
                print("Not enough steps to plot path with step sizes.")
                return fig, ax

            # Extract coordinates
            coords = np.array(self.coords_history)
            x = coords[:, 0]
            y = coords[:, 1]

            # Compute step sizes (Euclidean distance between consecutive points)
            step_sizes = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
            
            # Normalize step sizes for color mapping
            norm = Normalize(vmin=np.min(step_sizes), vmax=np.max(step_sizes))
            cmap = cm.viridis  # You can choose any other colormap

            # Create segments for LineCollection
            segments = [((x[i], y[i]), (x[i+1], y[i+1])) for i in range(len(x)-1)]

            # Create LineCollection
            lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2)
            lc.set_array(step_sizes)
            lc.set_zorder(3)
            ax.add_collection(lc)

            # Add a colorbar for step sizes
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, label='Step Size')

            # Optionally, plot markers
            ax.scatter(x, y, c=step_sizes, cmap=cmap, norm=norm, s=16, zorder=4, label='Path Steps')

            ax.legend()

            if show: 
                plt.show()
            if return_fig: 
                return fig, ax
