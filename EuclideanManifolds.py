import numpy as np
import scipy.optimize as opt
from shapely.geometry import Point, Polygon, GeometryCollection
from shapely.plotting import plot_polygon
from shapely import affinity, empty, GeometryType
from abc import ABC, abstractmethod
import ipywidgets as widgets
from IPython.display import display

import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True)
mpl.rc('ytick', direction='in', right=True)


class ManifoldBase(ABC):
    """Base for compact manifolds."""
    
    _name = None
    _texname = None # Optional
    
    def __init__(self):
        self._BD = None
        if self._name is None:
            raise ValueError("_name must be set")

    @abstractmethod
    def construct_base_domain(self):
        pass
    
    @abstractmethod
    def get_regions(self, r):
        pass
    
    @abstractmethod
    def set_lengths(self, *lengths):
        pass
    
    def make_geometry(self, r):
        a = GeometryCollection()
        for region in self.get_regions(r):
            if region.is_empty:
                continue
            a = a.union(region)
        a = self._BD.intersection(a)
        return a
    
    def area_ratio(self, r, show_plot=False, ax=None):
        """
        Calculate the ratio of areas for regions of radius r in the base domain.
        Inputs:
          r: float: radius of regions for the topology
          show_plot: bool: whether to show a plot of base domain with regions
                only use for testing/visualization, never for real calculations!
        Outputs:
          ratio: float: ratio of areas
        """
        a = self.make_geometry(r)
        if a.is_empty:
            return 0
            
        if show_plot:
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_aspect('equal')
            plot_polygon(a, ax=ax, add_points=False)
            plot_polygon(self._BD, ax=ax, add_points=False, color='c', alpha=0.5)
            print(a.area, self._BD.area, a.area / self._BD.area)
        return a.area / self._BD.area
    
    def get_name(self):
        """Return the name of the manifold as a string."""
        return self._name
    
    def get_texname(self):
        """Return the name of the manifold in TeX if available, otherwise return the name."""
        return self._texname or self.get_name()


class Orientable(ManifoldBase):
    """Base for compact orientable manifolds."""

    _LBmin = None
    
    def __init__(self, resolution=100):
        super().__init__()
        if self._LBmin is None:
            raise ValueError("_LBmin must be set")
        self.resolution = resolution
        
    def LB_from_radius(self, r, usemin=False):
        """
        Calculate the length, LB, perpendicular to the xy-plane for a given radius.
        The minimum allowed LB for the topology such that all observers see circles
        in the CMB can be enforced.
        Inputs:
          r: float: radius
          usemin: boolean: default=False
            whether to require LB no smaller than the topology minimum
        Outputs:
          LB: float: Length perpendicular to the xy-plane.
              LB will always be >= 0.
        """
        lb2 = self.LB_squared_from_radius(r)
        lb = np.sqrt(max(lb2, 0))
        if usemin:
            lb = max(self._LBmin, lb)
        return lb

    def radius_from_LB(self, LB, usemin=False):
        """
        Calculate the radius of the "main" circle from LB, the length perpendicular to the xy-plane.
        The minimum allowed LB for the topology such that all observers see circles
        in the CMB can be enforced.
        Inputs:
          LB: float: radius
          usemin: boolean: default=False
            whether to require LB no smaller than the topology minimum
        Outputs:
          radius: float: Radius of the main circle.
        """
        if usemin:
            LB = max(self._LBmin, LB)
        return np.sqrt(self.radius_squared_from_LB(LB))

    def LB_from_fraction(self, LA, frac):
        """
        Calculate LB for LA such that the fraction p of observers that see circles
        in the CMB.
        Inputs:
          LA: float or 1d-array: Length of the base domain in the xy-plane
          frac: float: fraction of observers that see circles in the CMB
        Outpus:
          LB: 1d-array: Length perpendicular to the xy-plane for each LA.
              An array with the same number of entries as LA is always returned,
              a one element array is returned for a single input LA.
        """
        LA = np.atleast_1d(LA)
        LB = np.zeros_like(LA)
        for j, la in enumerate(LA):
            if la <= 1:
                LB[j] = 2
                continue
            self.set_LA(la)
            r = opt.brentq(lambda x: frac - self.area_ratio(x, show_plot=False), 0, la)
            LB[j] = self.LB_from_radius(r, usemin=True)
        return LB


class Nonorientable(ManifoldBase):
    """Base for compact nonorientable manifolds."""
    
    def __init__(self):
        super().__init__()

    # CHANGE THIS
    def radius_from_LB(self, lb):
        return lb

class ManifoldE2(Orientable):
    _LBmin = 1 / 2
    _name = "E2"
    _texname = r'$E_2$'
    
    def set_lengths(self, *lengths):
        """E2 can have 1, 2, or 3 lengths:
        LA
        L1, L2
        L1, L2x, L2y
        """
        match len(lengths):
            case 1:
                self.lengths = [lengths[0], 0, lengths[0]]
            case 2:
                self.lengths = [lengths[0], 0, lengths[1]]
            case 3:
                self.lengths = [*lengths]
            case _:
                raise ValueError(f"E2 base is defined by 1, 2, or 3 lengths, not {len(lengths)}")
        self.TA1 = np.array([self.lengths[0], 0])
        self.TA2 = np.array(self.lengths[1:])
        self.construct_base_domain()
        
    def construct_base_domain(self):
        self._BD = Polygon([(0, 0), self.TA1, self.TA1 + self.TA2, self.TA2])
    
    def LB_squared_from_radius(self, r):
        return 1 - 4 * r**2
    
    def radius_squared_from_LB(self, LB):
        return (1 - LB**2) / 4
    
    def get_regions(self, r):
        centers = [(0, 0), self.TA1/2, self.TA1,
                   self.TA1 + self.TA2/2,
                   self.TA1 + self.TA2,
                   self.TA1/2 + self.TA2,
                   self.TA2, self.TA2/2,
                   self.TA1/2 + self.TA2/2,
                  ]
        return [Point(center).buffer(r, resolution=self.resolution)
                for center in centers]

class ManifoldE3(Orientable):
    _LBmin = 1 / 4
    _name = "E3"
    _texname = r'$E_3$'
    
    def set_lengths(self, LA):
        self.LA = LA
        self.construct_base_domain()
        
    def construct_base_domain(self):
        self._BD = Polygon([(0, 0), (self.LA, 0), (self.LA, self.LA), (0, self.LA)])
    
    def LB_squared_from_radius(self, r):
        return 1 - 2 * r**2
    
    def radius_squared_from_LB(self, LB):
        return (1 - LB**2) / 2
    
    def get_regions(self, r):
        centers = [(0, 0), (self.LA, 0), (self.LA, self.LA),
                   (0, self.LA), (self.LA/2, self.LA/2)]
        centers2 = [(self.LA/2, 0), (self.LA, self.LA/2),
                    (self.LA/2, self.LA), (0, self.LA/2)]
        circles= [Point(center).buffer(r, resolution=self.resolution)
                for center in centers]
        if self.LB_from_radius(r) < 1 / 2:
            r2 = np.sqrt(2*r**2 - 3/4)
            circles.extend([Point(center).buffer(r2, resolution=self.resolution)
                           for center in centers2])
        return circles

class ManifoldE4(Orientable):
    _LBmin = 1 / 3
    _name = "E4"
    _texname = r'$E_4$'
    
    def set_lengths(self, LA):
        self.LA = LA
        self.construct_base_domain()
        
    def construct_base_domain(self):
        self._BD = Polygon([(0, 0), (self.LA, 0),
                            (self.LA/2, np.sqrt(3)*self.LA/2),
                            (-self.LA/2, np.sqrt(3)*self.LA/2)])
    
    def LB_squared_from_radius(self, r):
        return 1 - 3 * r**2
    
    def radius_squared_from_LB(self, LB):
        return (1 - LB**2) / 3
    
    def get_regions(self, r):
        E4_centers = [(0, 0), (self.LA, 0), (self.LA/2, np.sqrt(3)*self.LA/2),
                      (-self.LA/2, np.sqrt(3)*self.LA/2),
                      (self.LA/2, self.LA/(2*np.sqrt(3))), (0, self.LA/np.sqrt(3))]
        return [Point(center).buffer(r, resolution=self.resolution)
                for center in E4_centers]

class ManifoldE5(Orientable):
    _LBmin = 1 / 6
    _name = "E5"
    _texname = r'$E_5$'
        
    def set_lengths(self, LA):
        self.LA = LA
        self.construct_base_domain()
        
    def construct_base_domain(self):
        self._BD = Polygon([(0, 0), (self.LA, 0),
                            (self.LA/2, np.sqrt(3)*self.LA/2),
                            (-self.LA/2, np.sqrt(3)*self.LA/2)])
    
    def LB_squared_from_radius(self, r):
        return 1 - r**2

    def radius_squared_from_LB(self, LB):
        return 1 - LB**2

    def get_regions(self, r):
        E5_centers = [(0, 0), (self.LA, 0), (self.LA/2, np.sqrt(3)*self.LA/2),
                      (-self.LA/2, np.sqrt(3)*self.LA/2)]
        E5_centers2 = [(self.LA/2, self.LA/(2*np.sqrt(3))), (0, self.LA/np.sqrt(3))]
        E5_centers3 = [(self.LA/2, 0), (self.LA/4, np.sqrt(3)*self.LA/4),
                       (3*self.LA/4, np.sqrt(3)*self.LA/4),
                       (-self.LA/4, np.sqrt(3)*self.LA/4), (0, np.sqrt(3)*self.LA/2)]

        circles= [Point(center).buffer(r, resolution=self.resolution)
                  for center in E5_centers]
        if (lb := self.LB_from_radius(r)) < 1 / 2:
            r2 = np.sqrt(4 * r**2 / 3 - 1)
            circles.extend([Point(center).buffer(r2, resolution=self.resolution)
                            for center in E5_centers2])
            if lb < 1 / 3:
                r3 = np.sqrt((9 * r**2 - 8) / 4)
                circles.extend([Point(center).buffer(r3, resolution=self.resolution)
                                for center in E5_centers3])
                
        return circles

class ManifoldE7(Nonorientable):
    _name = "E7"
    _texname = r'$E_7$'
        
    def set_lengths(self, L1y, L2x, L2z):
        self.L1y = L1y
        self.L2x = L2x
        self.L2z = L2z
        self.construct_base_domain()
        
    def construct_base_domain(self):
        self._BD = Polygon([(0, -self.L1y/2), (1, -self.L1y/2),
                            (1, self.L1y/2),
                            (0, self.L1y/2)])
    
    def get_regions(self, LAx):
        # L2x can be shifted by integer multiples of 2 LAx.
        # To find the closest clone we first enforce |L2x| < LAx
        L2x = self.L2x
        while L2x > LAx:
            L2x -= 2 * LAx
        while L2x < -LAx:
            L2x += 2 * LAx
        # Then we find the smallest values for (LAx - L2x)^2 allowing for shifts
        # by +/- 2 LAx.
        dL2x = np.min([(LAx - L2x)**2, (3*LAx - L2x)**2, (LAx + L2x)**2])
        dy1=np.sqrt(max(0, 1 - LAx**2))/2
        dy2=np.sqrt(max(0, 1 - dL2x - self.L2z**2))/2
        
        # Regions are rectangles
        if (dy := max(dy1, dy2)) > 0:
            rects = [
                Polygon([(0, -dy), (1, -dy), (1, dy), (0, dy)]),
                Polygon([(0, -self.L1y/2 + dy), (1, -self.L1y/2 + dy), (1, -self.L1y/2), (0, -self.L1y/2)]),
                Polygon([(0, self.L1y/2 - dy), (1, self.L1y/2 - dy), (1, self.L1y/2), (0, self.L1y/2)]),
            ]
        else:
            rects = empty(1, geom_type=GeometryType.POLYGON).tolist()
        return rects

# E9 is very, very similar to E7
class ManifoldE9(ManifoldE7):
    _name = "E9"
    _texname = r'$E_9$'
        
    def get_regions(self, LAx):
        # L2x can be shifted by integer multiples of 2 LAx.
        # To find the closest clone we first enforce |L2x| < LAx
        L2x = self.L2x
        while L2x > LAx:
            L2x -= 2 * LAx
        while L2x < -LAx:
            L2x += 2 * LAx
        # Then we find the smallest values for (LAx - L2x)^2 allowing for shifts
        # by +/- 2 LAx.
        dL2x = np.min([(LAx - L2x)**2, (3*LAx - L2x)**2, (LAx + L2x)**2])
        dy1=np.sqrt(max(0, 1 - LAx**2))/2
        dy2=np.sqrt(max(0, 1 - dL2x - self.L2z**2))/2

        rects = []
        if dy1 > 0:
            rects.extend([
                Polygon([(0, -dy1), (1, -dy1), (1, dy1), (0, dy1)]),
                Polygon([(0, -self.L1y/2 + dy1), (1, -self.L1y/2 + dy1), (1, -self.L1y/2), (0, -self.L1y/2)]),
                Polygon([(0, self.L1y/2 - dy1), (1, self.L1y/2 - dy1), (1, self.L1y/2), (0, self.L1y/2)]),
            ])
        if dy2 > 0:
            rects.extend([
                Polygon([(0, -self.L1y/4 + dy2), (1, -self.L1y/4 + dy2), (1, -self.L1y/4 - dy2), (0, -self.L1y/4) - dy2]),
                Polygon([(0, self.L1y/4 - dy2), (1, self.L1y/4 - dy2), (1, self.L1y/4 + dy2), (0, self.L1y/4 + dy2)]),
            ])
        if len(rects) == 0:
            rects = empty(1, geom_type=GeometryType.POLYGON).tolist()
        return rects

class ManifoldE8(Nonorientable):
    _name = "E8"
    _texname = r'$E_8$'

    def __init__(self, resolution=100):
        super().__init__()
        self.resolution = resolution
        
    def set_lengths(self, LCy, LBz):
        self.LCy = LCy
        self.LBz = LBz
        self.construct_base_domain()
        
    def construct_base_domain(self):
        self._BD = Polygon([(0, -self.LCy/2), (1, -self.LCy/2),
                            (1, self.LCy/2),
                            (0, self.LCy/2)])
    
    def get_regions(self, LAx):
        # Regions are rectangles
        dy = np.sqrt(max(0, 1 - LAx**2)) / 2
        # Recall that x is measured in units of LAx so our circle becomes an ellipse
        dx = np.sqrt(max(0, 1 - self.LBz**2)) / (2 * LAx)
        regions = []
        if dy > 0:
            regions.extend([Polygon([(0, -dy), (1, -dy), (1, dy), (0, dy)]),
            Polygon([(0, -self.LCy/2 + dy), (1, -self.LCy/2 + dy), (1, -self.LCy/2), (0, -self.LCy/2)]),
            Polygon([(0, self.LCy/2 - dy), (1, self.LCy/2 - dy), (1, self.LCy/2), (0, self.LCy/2)]),
        ])
        if dx > 0:
            centers = [(1/2, 0), (1/2, self.LCy/2), (1/2, -self.LCy/2),]
            xscale = dx
            yscale = dx * LAx
            regions.append(Polygon([(0, -self.LCy/2), (0, self.LCy/2), (dx, self.LCy/2), (dx, -self.LCy/2)]))
            regions.append(Polygon([(1, -self.LCy/2), (1, self.LCy/2), (1-dx, self.LCy/2), (1-dx, -self.LCy/2)]))
            regions.extend([affinity.scale(Point(center).buffer(1, resolution=self.resolution), xfact=xscale, yfact=yscale)
                                for center in centers])
        return regions

# E10 is very, very similar to E8
class ManifoldE10(ManifoldE8):
    _name = "E10"
    _texname = r'$E_{10}$'

    def get_regions(self, LAx):
        # Regions are rectangles
        dy = np.sqrt(max(0, 1 - LAx**2)) / 2
        regions = []
        if dy > 0:
            regions.extend([Polygon([(0, -dy), (1, -dy), (1, dy), (0, dy)]),
            Polygon([(0, -self.LCy/2 + dy), (1, -self.LCy/2 + dy), (1, -self.LCy/2), (0, -self.LCy/2)]),
            Polygon([(0, self.LCy/2 - dy), (1, self.LCy/2 - dy), (1, self.LCy/2), (0, self.LCy/2)]),
        ])
        # Recall that x is measured in units of LAx so our circle becomes an ellipse
        dx2 = np.sqrt(max(0, 1 - self.LBz**2 - self.LCy**2/4)) / (2 * LAx)
        if dx2 > 0:
            regions.append(Polygon([(0, -self.LCy/2), (0, self.LCy/2), (dx2, self.LCy/2), (dx2, -self.LCy/2)]))
            regions.append(Polygon([(1, -self.LCy/2), (1, self.LCy/2), (1-dx2, self.LCy/2), (1-dx2, -self.LCy/2)]))
        dx1 = np.sqrt(max(0, 1 - self.LBz**2)) / (2 * LAx)
        if dx1 > 0:
            centers = [(1/2, self.LCy/4), (1/2, -self.LCy/4),]
            xscale = dx1
            yscale = dx1 * LAx
            regions.extend([affinity.scale(Point(center).buffer(1, resolution=self.resolution), xfact=xscale, yfact=yscale)
                                for center in centers])
        return regions

class ManifoldPlotter:
    _implemented_manifold_names = ['E2square', 'E2rectangle', 'E2',
                                   'E3', 'E4', 'E5', 'E7', 'E8', 'E9', 'E10']
    
    def __init__(self, cmap=None, manifold_name=None):
        self.window = None
        self.fig = None
        self.cmap = cmap or mpl.colormaps['viridis']
        with plt.ioff():
            self.fig = plt.figure(figsize=(6, 2.5), layout='constrained')
        self.fig.canvas.toolbar_visible = False
        self.fig.canvas.header_visible = False
        self.ax = self.fig.add_subplot(111)

        self.length_sliders = []
        # Default lengths
        self.LBarr = [0.3, 0.4, 0.45, 0.5, 0.6, 0.8, 0.9, 0.98]

        self._make_widgets()
        display(self.window)
        # Default to first manifold
        self.setup_manifold(manifold_name or self._implemented_manifold_names[0])

    # Only setup the plot? Do not CREATE the window?
    def setup_manifold(self, manifold_name:str):
        self._clear_length_sliders()
        self.LB_label.value = r'$L_{B}\ $ (add or remove values below)'
        self.legend_title = r'$L_B/L_{\mathrm{LSS}}$ (frac)'
        self.ax.cla()
        self.ax.set_xlabel(r'$x/L_{\mathrm{LSS}}$');
        self.ax.set_ylabel(r'$y/L_{\mathrm{LSS}}$')
        self.ax.set_aspect('equal')
        # Not completely sure why this is needed ....
        try:
            # Defaults. Can be overridden below.
            self.LB_input.value = [0.3, 0.4, 0.45, 0.5, 0.6, 0.8, 0.9, 0.98]
        except NotImplementedError:
            pass

        # Defaults. Can be overridden below.
        self.LB_input.min = 0
        self.LB_input.max = 1

        match manifold_name:
            case 'E2square':
                self.manifold = ManifoldE2()
                LA = 2
                s = self.length_sliders[0]
                s.min = 1
                s.max = 4
                s.step = 0.1
                s.value = LA
                s.disabled = False
                s.layout.visibility = 'visible'
                s.description='$L_A$'
                s.observe(self._on_length_change, names='value')
                self.infobox.value=r"""$E_2$: half-turn space, with square base.
                Regions that see clones are cylinders with radii determined by $0 < L_B < 1$.
                Shown is a square base of this domain with the length $L_A$ adjustable.
                """
            case 'E2rectangle':
                self.manifold = ManifoldE2()
                L1 = 2
                L2 = 2
                # Shorthand names for the sliders.
                s1, s2 = self.length_sliders[:2]
                s1.min = 1
                s1.max = 4
                s1.step = 0.1
                s1.value = L1
                s1.disabled = False
                s1.layout.visibility = 'visible'
                s1.description='$L_1$'
                s1.observe(self._on_length_change, names='value')
                s2.min = 1
                s2.max = 6
                s2.step = 0.1
                s2.value = L2
                s2.disabled = False
                s2.layout.visibility = 'visible'
                s2.description='$L_2$'
                s2.observe(self._on_length_change, names='value')
                self.infobox.value=r"""$E_2$: half-turn space, with rectangular base.
                Regions that see clones are cylinders with radii determined by $0 < L_B < 1$.
                Shown is a rectangular base of this domain with the lengths $L_1$ along the $x$-axis and $L_2$ along the $y$-axis adjustable.
                """
            case 'E2':
                self.manifold = ManifoldE2()
                L1 = 2
                L2x = 0
                L2y = 2
                # Shorthand names for the sliders.
                s1, s2x, s2y = self.length_sliders[:3]

                s1.min = 1
                s1.max = 4
                s1.value = L1
                s1.disabled = False
                s1.layout.visibility = 'visible'
                s1.description='$L_1$'
                s1.observe(self._on_length_change, names='value')
                s2x.min = -L1/2
                s2x.max = L1/2
                s2x.step = 0.05
                s2x.value = L2x
                s2x.disabled = False
                s2x.layout.visibility = 'visible'
                s2x.description='$L_{2x}$'
                s2x.observe(self._on_length_change, names='value')
                s2y.max = 6 # Set first so that max > min.
                s2y.min = np.sqrt(L1**2 - L2x**2)
                s2y.step = 0.1
                s2y.value = L2y
                s2y.disabled = False
                s2y.layout.visibility = 'visible'
                s2y.description='$L_{2y}$'
                s2y.observe(self._on_length_change, names='value')
                # Extra observers to reset the min and max.
                def _update_L2x_slider(change):
                    newval = s1.value / 2
                    s2x.min = -newval
                    s2x.max = newval
                def _update_L2y_slider(change):
                    newval = np.sqrt(s1.value**2 - s2x.value**2)
                    s2y.min = newval
                s1.observe(_update_L2y_slider, names='value')
                s1.observe(_update_L2x_slider, names='value')
                s2x.observe(_update_L2y_slider, names='value')
                self.infobox.value=r"""$E_2$: half-turn space, with parallelogram base.
                Regions that see clones are cylinders with radii determined by $0 < L_B < 1$.
                Shown is the most general parallelogram base of this domain with the lengths $L_1$ along the $x$-axis and $L_{2x}$ and $L_{2y}$ for the other side of the parallelogram adjustable.
                The lengths are constrained to ensure the smallest base domain. This means $-L_1/2 < L_{2x} < L_1/2$ and $L_{2y} > \sqrt{L_1^2 - L_{2x}^2}$.
                """
            case 'E3' | 'E4' | 'E5':
                LA = 2
                s = self.length_sliders[0]
                s.min = 1
                s.max = 4
                s.step = 0.1
                s.value = LA
                s.disabled = False
                s.layout.visibility = 'visible'
                s.description='$L_A$'
                s.observe(self._on_length_change, names='value')
                if manifold_name == 'E3':
                    self.manifold = ManifoldE3()
                    self.infobox.value=r"""$E_3$: quarter-turn space has a square base.
                Regions that see clones are cylinders with radii determined by $0 < L_B < 1$.
                The square base of this domain has side lengths $L_A$ adjustable.
                """
                elif manifold_name == 'E4':
                    self.manifold = ManifoldE4()
                    self.infobox.value=r"""$E_4$: third-turn space has a rhombus base.
                Regions that see clones are cylinders with radii determined by $0 < L_B < 1$.
                The rhombus base of this domain has side lengths $L_A$ adjustable.
                """
                else:
                    self.manifold = ManifoldE5()
                    self.infobox.value=r"""$E_5$: sixth-turn space has a rhombus base.
                Regions that see clones are cylinders with radii determined by $0 < L_B < 1$.
                The rhombus base of this domain has side lengths $L_A$ adjustable.
                """
            case 'E7' | 'E9':
                if manifold_name == 'E7':
                    self.manifold = ManifoldE7()
                elif manifold_name == 'E9':
                    self.manifold = ManifoldE9()
                # Not completely sure why this is needed ....
                try:
                    # Less than 1/2 always sees circles.
                    self.LB_input.value = [0.5, 0.6, 0.8, 0.9, 0.98, 1.2]
                except NotImplementedError:
                    pass
                self.LB_input.min = 1 / 2
                self.LB_input.max = 1.8
                self.LB_label.value = r'$L_{Ax}\ $' f' (add or remove values below, max = {self.LB_input.max})'
                self.legend_title = r'$L_{Ax}/L_{\mathrm{LSS}}$ (frac)'
                self.ax.set_xlabel(r'$x/L_{Ax}$')
                self.ax.set_aspect('auto')
                L1y = 4.5
                L2x = 0
                L2z = 1
                # Shorthand names for the sliders.
                s1y, s2x, s2z = self.length_sliders[:3]

                s1y.min = 1
                s1y.max = 8
                s1y.value = L1y
                s1y.disabled = False
                s1y.layout.visibility = 'visible'
                s1y.description='$L_{1y}$'
                s1y.observe(self._on_length_change, names='value')
                s2x.min = -1
                s2x.max = 1
                s2x.step = 0.05
                s2x.value = L2x
                s2x.disabled = False
                s2x.layout.visibility = 'visible'
                s2x.description='$L_{2x}$'
                s2x.observe(self._on_length_change, names='value')
                s2z.max = 2
                s2z.min = 1
                s2z.step = 0.05
                s2z.value = L2z
                s2z.disabled = False
                s2z.layout.visibility = 'visible'
                s2z.description='$L_{2z}$'
                s2z.observe(self._on_length_change, names='value')
                # Extra observers to reset the min and max.
                def _update_L2z_slider(change):
                    newval = np.sqrt(max(0.01, 1 - s2x.value**2))
                    s2z.min = newval
                s2x.observe(_update_L2z_slider, names='value')
                if manifold_name == 'E7':
                    self.infobox.value=r"""$E_7$: Klein space.
                    Regions that see clones are rectangles with height determined by $L_{Ax}$ or if $L_{2x} \neq 0$ then $(L_{Ax}-L_{2x})^2 + L_{2z}^2$."""
                elif manifold_name == 'E9':
                    self.infobox.value=r"""$E_9$: Klein space with vertical flip.
                    Regions that see clones are rectangles with height determined by $L_{Ax}$ and may include rectangles with height determined by $(L_{Ax}-L_{2x})^2 + L_{2z}^2$."""
            case 'E8' | 'E10':
                if manifold_name == 'E8':
                    self.manifold = ManifoldE8()
                elif manifold_name == 'E10':
                    self.manifold = ManifoldE10()
                # Less than 1/2 always sees circles
                # Not completely sure why this is needed ....
                try:
                    # Less than 1/2 always sees circles.
                    self.LB_input.value = [0.5, 0.6, 0.8, 0.9, 0.98, 1.2]
                except NotImplementedError:
                    pass
                self.LB_input.min = 1 / 2
                self.LB_input.max = 1.8

                self.LB_label.value = r'$L_{Ax}\ $ ' f' (add or remove values below, max = {self.LB_input.max})'
                self.legend_title = r'$L_{Ax}/L_{\mathrm{LSS}}$ (frac)'
                self.ax.set_xlabel(r'$x/L_{Ax}$')
                self.ax.set_aspect('auto')
                LCy = 3.5
                LBz = 0.9
                # Shorthand names for the sliders.
                sCy, sBz = self.length_sliders[:2]
                
                sCy.min = 1
                sCy.max = 8
                sCy.value = LCy
                sCy.disabled = False
                sCy.layout.visibility = 'visible'
                sCy.description = r'$L_{Cy}$'
                sCy.observe(self._on_length_change, names='value')
                sBz.max = 2
                sBz.min = 1 / 2
                sBz.step = 0.01
                sBz.value = LBz
                sBz.disabled = False
                sBz.layout.visibility = 'visible'
                sBz.description='$L_{Bz}$'
                sBz.observe(self._on_length_change, names='value')
                # Extra observers to reset the min and max.
                if manifold_name == 'E8':
                    self.infobox.value=r"""$E_8$: Klein space with horizontal flip.
                    Regions that see clones are rectangles and possibly circles."""
                elif manifold_name == 'E10':
                    self.infobox.value=r"""$E_{10}$: Klein space with half-turn.
                    Regions that see clones are rectangles and possibly circles."""
            case _:
                raise ValueError(f'Unknown manifold name: "{manifold_name}"')

        self.ax.set_title(self.manifold.get_texname())

        self._set_manifold_lengths()
        self.patches = []
        self.outlines = []
        self.fill_plot()
        self._update_figure()

    def _set_manifold_lengths(self):
        lengths = [s.value for s in self.length_sliders if not s.disabled]
        self.manifold.set_lengths(*lengths)

    def _on_length_change(self, change):
        self._set_manifold_lengths()
        self._update_figure()
        
    def _on_LB_change(self, change):
        if len(self.LBarr) != len(change.new):
            # Keep the list sorted, this looks better.
            self.LBarr = list(sorted(change.new))
            self.LB_input.value = self.LBarr
            self._update_figure()

    def _update_figure(self):
        # Remove all old patches
        for a in self.patches:
            a.remove()
        for a in self.outlines:
            a.remove()
        self.patches = []
        self.outlines = []
        # Refill with patches
        self.fill_plot()
        
    def _clear_length_sliders(self):
        """Hide all sliders, remove all handlers."""
        for s in self.length_sliders:
            s.disabled = True
            s.layout.visibility = 'hidden'
            s.unobserve_all()
            
    def _make_widgets(self):
        """Makes self.window. Does not display the widget!"""
        LB_layout = widgets.Layout(width='90%',
                                   border='2px solid', 
                                   align_content='flex-start')
        self.LB_input = widgets.FloatsInput(description=r'$L_B$:',
                                            value=self.LBarr,
                                            allow_duplicates=False,
                                            min=0, max=1,
                                            layout=LB_layout,
                                            format='.2f')
        self.LB_label = widgets.Label(r'$L_{B}$ (add or remove values below)')
        # This needs work
        LBwidget = widgets.VBox([self.LB_label, self.LB_input])
        self.LB_input.observe(self._on_LB_change, names='value')
        # Widgets for other length inputs. All may not be needed.
        # These MUST be set/turned on and off by manifolds.
        L1_slider = widgets.FloatSlider(value=0, min=0, max=1, step=0.1,
                                        description=r'$L_1$',
                                        disabled=True,
                                        orientation='horizontal',
                                        continuous_update=False,
                                        readout=True)
        L2_slider = widgets.FloatSlider(value=0, min=0, max=1, step=0.05,
                                         description=r'$L_2$',
                                         disabled=True,
                                         orientation='horizontal',
                                         continuous_update=False,
                                         readout=True)
        L3_slider = widgets.FloatSlider(value=0, min=0, max=1, step=0.1,
                                         description=r'$L_2$',
                                         disabled=True,
                                         orientation='horizontal',
                                         continuous_update=False,
                                         readout=True)
        self.length_sliders = [L1_slider, L2_slider, L3_slider]
        lengthsliders = widgets.VBox(self.length_sliders)
        manifoldchooser = widgets.Dropdown(
            options=self._implemented_manifold_names,
            value=self._implemented_manifold_names[0],
            description='Manifold:',
            disabled=False)
        manifoldchooser.observe(lambda change: self.setup_manifold(change.new), names='value')
        sidebar = widgets.VBox([manifoldchooser, lengthsliders, LBwidget])
        self.infobox = widgets.HTMLMath(
            value='Manifold Info',
            disabled=True
        )
        self.window = widgets.AppLayout(header=None,
                                        left_sidebar=sidebar,
                                        center=self.fig.canvas,
                                        right_sidebar=None,
                                        footer=self.infobox,
                                        pane_widths=[2.4, 5, 0]
                                       )
        
    def fill_plot(self):
        cmin = min(self.LBarr)
        cmax = max(self.LBarr)
        color_mapper = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
        (xmin, ymin, xmax, ymax) = self.manifold._BD.bounds
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        # LBarr must be sorted!
        for lb in self.LBarr:
            # Radius should not be used here, just pass lb to the manifold.
            r = self.manifold.radius_from_LB(lb)
            if (frac := self.manifold.area_ratio(r)) > 0:
                g = self.manifold.make_geometry(r)
            else:
                # Empty polygon so fake it by creating one off screen.
                g = Polygon([(xmin-1, ymin-1), (xmin-1, ymin-2), (xmin-2, ymin-2)])
            self.patches.append(plot_polygon(g, ax=self.ax, color=self.cmap(color_mapper(lb)), add_points=False, alpha=1,
                        label=f'{lb:g} ({100*frac:2.0f}%)'))
        # Now do it again and draw an outline "on top" in case some of the nested regions fully cover others.
        for lb in self.LBarr:
            # Radius should not be used here, just pass lb to the manifold.
            r = self.manifold.radius_from_LB(lb)
            if (frac := self.manifold.area_ratio(r)) > 0:
                g = self.manifold.make_geometry(r)
            else:
                # Empty polygon so skip
                continue
            self.outlines.append(plot_polygon(g, ax=self.ax, edgecolor=self.cmap(color_mapper(lb)), facecolor='none', add_points=False, fill=False, lw=2))
        self.patches.append(plot_polygon(self.manifold._BD, ax=self.ax, add_points=False, lw=2, color='k', fill=False))
        self.ax.legend(loc='center left',
                       title=self.legend_title,
                       bbox_to_anchor=(1.01, 0.5), bbox_transform=self.ax.transAxes,
                       ncols=len(self.LBarr)//9 + 1,
                      )
