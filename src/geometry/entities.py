import numpy as np

from src.geometry.utils import get_rotation_matrix, get_pos_point_on_segment

class Line:

    def __init__(self, vecdir, intersept):

        self.vecdir = vecdir / np.linalg.norm(vecdir)
        self.intersept = intersept

    def is_inside(self, u):
        return (np.cross(self.vecdir - self.intersept, u) == 0).all()

    def get_coords(self, t):
        return self.vecdir * t + self.intersept

    def plot_on_ax(self, ax, t1, t2, c='r', **kwargs):

        x1 = self.get_coords(t1)
        x2 = self.get_coords(t2)
        ax.plot([x1[0], x2[0]], [x1[1], x2[1]], [x1[2], x2[2]], c=c, **kwargs)

class Plane:

    def __init__(self, vecn, intersept=None, point=None, vecs=[]):
        """
        <vecn . x > = intersept, for all x in plane
        Args:
            vecn (nd array): Normal vector
            intersept ([type]): intersept
            vecs (list, optional): Other vectors of the basis. Defaults to [].
        """
        self.vecn = vecn / np.linalg.norm(vecn)
        self.R = get_rotation_matrix(vecs + [vecn])
        if point is not None:
            self.intersept = self.vecn @ point
        else:
            self.intersept = intersept if intersept is not None else 0


    def __repr__(self):
        return 'Plane equation: {}x + {}y + {}z = {}'.format(*np.round(self.vecn, 3), np.round(self.intersept, 3))

    def is_inside(self, u):
        if not isinstance(u, Line):
            return u @ self.vecn == self.intersept

        if u.vecdir @ self.vecn != 0:
            return False
        return u.intersept @ self.vecn == 0
        

    def is_line_orthogonal(self, line):
        return (np.cross(self.vecn, line.vecdir) == 0).all()
    
    def is_line_parallel(self, line):
        return self.vecn @ line.vecdir == 0

    def intersept_line(self, line):
        """
        Returns the coordinates of the intersection of the line with the plane
        
        Args:
            line (Line child): Line object which intersepts the plane
        
        Returns:
            ndarray: coordinates (3,) of the point
        """
        assert not self.is_line_parallel(line), 'line is parallel to plane'
        
        t = (self.intersept - self.vecn @ line.intersept) / (self.vecn @ line.vecdir)
        # print(t, self.intersept, line.intersept, self.vecn, line.vecdir)
        return line.get_coords(t)

    def sample(self, ns, size):
        """
        Samples a grid of points contained in the hyerplane.
        
        Args:
            ns (int): Number of points per dimension. The total number of points
                     will be n0 x n1.
            size (tuple): (width, length). Size of the rectangle of samples.
        
        Returns:
            (ndarrray, ndarray, ndarray): samples (3, n x n), 
                                        x coordinate (n x n), y coordinate (n x n)
        """
        if type(ns) == int:
            n0, n1 = ns, ns
        elif len(ns) == 2:
            n0, n1 = ns
        else:
            assert False, 'ns must be int or contain two ints'
        if type(size) == int:
            size = np.array([size, size])
        X, Y = np.mgrid[0:n0, 0:n1]
        samples = np.c_[X[X == X] * size[0] / (n0-1), Y[Y == Y] * size[1] / (n1-1)].T
        samples = np.vstack((samples, np.zeros(n0*n1)))
        samples = self.R @ samples
        samples = samples + (self.vecn * self.intersept / (self.vecn @ self.vecn))[:, np.newaxis]

        return samples, X[X == X], Y[Y == Y]

    def sample_with_center(self, n, size, center):
        """
        Samples plane centered on center.
        
        Args:
            n (int): Number of points per dimension. The total number of points
                     will be n x n.
            size (tuple): (width, length). Size of the rectangle of samples.
            center (ndarray): center of the sampling
        
        Returns:
            (ndarrray, ndarray, ndarray): samples (3, n x n), 
                                        x coordinate (n x n), y coordinate (n x n)
        """
        samples, X, Y = self.sample(n, size)
        correction = (center - samples.mean(1))[:, np.newaxis]
        samples += correction
        return samples, X, Y

    
    def sample_between_coords(self, n, coords):
        """
        Samples plane between the coordinates coords. 
        
        Args:
            n (int): Number of points per dimension. The total number of points
                     will be n x n.
            coords (ndarray): shape (3, 4) or (4, 3). coordinates of the 4 points to sample between.
                              The 4 points must be given in a "circle" way (e.g. ABCD is a parallelogramme, not crossed)

        Returns:
            (ndarrray, ndarray, ndarray): samples (3, n x n), 
                                        x coordinate (n x n), y coordinate (n x n)
        """

        if coords.shape == (3,4):
            coords = coords.T
        P1, P2, P3, P4 = coords
        # Check if we have parallelogramme
        assert np.abs( np.linalg.norm(P1 - P2) - np.linalg.norm(P3 - P4) ) / np.linalg.norm(P1 - P2) < .01
        assert np.abs( np.linalg.norm(P3 - P2) - np.linalg.norm(P1 - P4) ) / np.linalg.norm(P3 - P2) < .01

        size = (np.linalg.norm(P1-P2), np.linalg.norm(P2-P3))
        center = coords.mean(0)

        return self.sample_with_center(n, size, center)

    def plot_on_ax(self, ax, size, center, c='blue', **kwargs):
        """
        Plots the plane on an ax.
        Args:
            ax (mpl_toolkits.mplot3d.axes3d.Axes3D): Ax on which to plot the plane
            size (tuple): (width, length). Size of the rectangle of samples.
            center (ndarray): center of the sampling
            
        Returns:
            (ndarrray, ndarray, ndarray): samples (3, n x n), 
                                        x coordinate (n x n), y coordinate (n x n)
        """
        samples, X, Y = self.sample_with_center(2, size, center)
        samples[:,np.array([-1, -2])] = samples[:,np.array([-2, -1])]



        for i in range(4):
            ax.plot(
                [samples[0, i], samples[0, (i+1)%4]],
                [samples[1, i], samples[1, (i+1)%4]],
                [samples[2, i], samples[2, (i+1)%4]],
                c=c, **kwargs)
        
        return samples, X, Y

class Cube:

    def __init__(self, planes_inf, planes_sup,):

        self.planes_inf = planes_inf
        self.planes_sup = planes_sup

    def is_inside(self, u, error=1e-6):

        for plane in self.planes_inf:
            if plane.vecn @ u < plane.intersept - error:
                return False
        for plane in self.planes_sup:
            if plane.vecn @ u > plane.intersept + error:
                return False
        return True 

    def intersept_line(self, line):
        for plane in self.planes_sup:
            if plane.is_line_parallel(line):
                print('paral')
                continue
            x = plane.intersept_line(line)
            if self.is_inside(x):
                x1 = x
                break

        for plane in self.planes_inf:
            if plane.is_line_parallel(line):
                continue
            x = plane.intersept_line(line)

            if self.is_inside(x):
                x2 = x
                break

        return x1, x2

        

class StraightCube(Cube):

    def __init__(self, size=(1, 1, 1), norms=(1, 1, 1), displacement=np.zeros(3)):
        self.dim = len(size)
        assert self.dim == 3
        self.planes_inf = []
        self.planes_sup = []
        self.size = np.array(size)
        self.displacement = displacement
        self.basis = np.eye(self.dim)
        self.norms = np.array(norms)

        for i in range(self.dim):
            R = self.basis[np.arange(i, i+self.dim)%self.dim].T
            self.planes_inf.append(Plane(
                vecn=R[2], intersept=displacement@R[-1], vecs=[R[0], R[1]]
                ))

            self.planes_sup.append(Plane(
                vecn=R[2], intersept=displacement@R[-1] + self.size[2-i], vecs=[R[0], R[1]]
                ))

    
    @property
    def centroid(self):
        return self.size / 2 + self.displacement


    def faces(self, idx):
        return self.planes_inf[2 - idx], self.planes_sup[2 - idx]

    def get_face_center(self, idx):
        idx = idx%3
        center = (self.basis[(idx+2)%self.dim]*self.size[(idx+2)%self.dim] + self.basis[(idx+1)%self.dim]*self.size[(idx+1)%self.dim]) / 2
        return center

    def get_face_size(self, idx):
        idx = idx%3
        if idx == 0:
            return self.size[1], self.size[2]
        if idx == 1:
            return self.size[2], self.size[0]
        if idx == 2:
            return self.size[0], self.size[1]

    def intersept_line(self, line):
        for i in range(len(self.size)):
            plane1, plane2 = self.faces(i)
            if plane1.is_line_parallel(line):
                continue
            x = plane1.intersept_line(line)
            if self.is_inside(x):
                x1 = x
                x2 = plane2.intersept_line(line)

                return x1, x2

    
    def sample_centered_plane(self, n, size, plane):
        """
        Args:
            n ([type]): [description]
            size ([type]): [description]
            plane (Plane): [description]
        
        Returns:
            [type]: [description]
        """
        samples, Xs, Ys = plane.sample(n, size)
        center_correction = (self.size / 2 - samples.mean(1))[:, np.newaxis]

        samples += center_correction
        return samples, Xs, Ys

    
    def sample_position_plane(self, position, n, plane, size=None):
        
        if size is None:
            size = np.sqrt(self.size[1]**2 + self.size[2]**2) + np.zeros(2)

        samples, Xs, Ys = self.sample_centered_plane(n, size, plane)
        line = Line(plane.vecn, self.centroid)
        x1, x2 = self.intersept_line(line)

        samples += (.5 * position * (x2 - x1))[:, np.newaxis]
        x1_mm = x1 * self.norms
        x2_mm = x2 * self.norms
        SliceLocation = 0.5 * position * np.linalg.norm(x2_mm - x1_mm)
        return samples, Xs, Ys, SliceLocation
        

    def get_pixelspacing_direction(self, vecdir):
        vecn = vecdir / np.linalg.norm(vecdir)
        return np.linalg.norm(vecn * self.norms)


    def sample_cube(self, n):
        samples = []
        for i in range(len(self.size)):
            face1, face2 = self.faces(i)
            size = self.size[i], self.size[(i+1)%3]
            center = self.get_face_center(i)
            samples += [ 
                face1.sample_with_center(n, size, center)[0],
                face2.sample_with_center(n, size, center + face2.vecn*self.size[-1])[0]
            ]
        return np.concatenate(samples, axis=1)

    
    def plot_on_ax(self, ax, c='blue', **kwargs):
        for i in range(len(self.size)):
            face1, face2 = self.faces(i)
            size = self.get_face_size(i)
            center = self.get_face_center(i)
            face1.plot_on_ax(ax, size, center, c=c, **kwargs)
            face2.plot_on_ax(ax, size, center + face2.vecn*self.size[i], c=c, **kwargs)


    def plot_centered_plane(self, ax, plane, color_cube='blue', color_plane='red', size=None, **kwargs):

        if size is None:
            size = np.sqrt(self.size[1]**2 + self.size[2]**2) + np.zeros(2)
        center = self.centroid 
        plane.plot_on_ax(ax, size, center, c=color_plane)
        self.plot_on_ax(ax, c=color_cube)


    def plot_plane_pos_on_ax(self, ax, position, plane, color_cube='blue', color_plane='red', size=None):

        if size is None:
            size = np.sqrt(self.size[1]**2 + self.size[2]**2) + np.zeros(2)
        line = Line(plane.vecn, self.centroid)
        x1, x2 = self.intersept_line(line)
        center = self.centroid + .5 * position * (x2 - x1)
        plane.plot_on_ax(ax, size, center, c=color_plane)
        self.plot_on_ax(ax, c=color_cube)


    def get_indexs_oriented_on_cube(self, v, Xs, Ys, Zs):
        """
        Get the position of the point v initially expressed in the basis Xs, Ys, Zs,
        in the basis of the cube. Arrays of shape (2,3) can also be expressed as iteratives
        of len 2 with elements being arrays of shape (3,).
                
        Args:
            v (ndarray): vector V in the cube inside Xs, Ys, Zs
            Xs (ndarray): shape (2, 3). The two vectors used to define the X coordinate of the cube. 
            Ys (ndarray): shape (2, 3). The two vectors used to define the Y coordinate of the cube. 
            Zs (ndarray): shape (2, 3). The two vectors used to define the Z coordinate of the cube. 
            
        Returns:
            ndarray: shape (3,), the indexes of v in the cube.
        """
        ti = get_pos_point_on_segment(v, Xs[0], Xs[1])
        tj = get_pos_point_on_segment(v, Ys[0], Ys[1])    
        tk = get_pos_point_on_segment(v, Zs[0], Zs[1])    
                
        if len(v) == 1:
            i = int(self.size[0]*ti)
            j = int(self.size[1]*tj)
            k = int(self.size[2]*tk)
            return np.array([i,j,k])

        i = (self.size[0] * ti).astype(int)
        j = (self.size[1] * tj).astype(int)
        k = (self.size[0] * tk).astype(int)
        return np.stack([i, j, k], 0)
        

    def get_pos_plane(self, plane):
        """
        Get the position of the plane relatively to the center of the cube.
        If the center of the plane is on the cube, the relative position is either 1
        or -1.
        
        Args:
            plane (Plane): plane to get the position of
        
        Returns:
            float: in [-1, 1] indicating position of the plane.
        """
        line = Line(vecdir=plane.vecn, intersept=self.centroid)
        x1, x2 = self.intersept_line(line)

        cent_plane = plane.intersept_line(line).astype(int)
        pos = 2*np.linalg.norm(cent_plane - self.centroid)/ np.linalg.norm(x1-x2)
        return pos
