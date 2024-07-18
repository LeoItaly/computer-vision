import cv2 
import numpy as np
import itertools as it
import scipy.optimize as coolNonLinOptimizer
import matplotlib as plt
import math

#Pi that converts from homogeneous to inhomogeneous coordinates
def Pi(points_homogenous):
    points_inhomogenous = points_homogenous[:-1] / points_homogenous[-1]
    return points_inhomogenous

#PiInv that converts from inhomogeneous to homogeneous coordinates
def PiInv(points_inhomogeneous):
    points_homogeneous=np.vstack((points_inhomogeneous, np.ones(points_inhomogeneous.shape[1])))
    return points_homogeneous

#Construction of 3D points to simulate the projection
def box3d(n: int):
    """
        Description:
    -The box3d function generates a set of 3D coordinates forming a box with lines through its edges and a cross through the middle.

        Parameters:
    - number of points in 3D to convert

    Returns:
    - 15xn 3D points (3, 15xn)
    """
    points = []
    N = tuple(np.linspace(-1, 1, n))
    for i, j in [(-1, -1), (-1, 1), (1, 1), (0, 0)]:
        points.extend(set(it.permutations([(i, )*n, (j, )*n, N])))
    return np.hstack(points)/2

#Returning the projection matrix
def getProjection_Matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray):

    """
        Parameters:
    - K: Camera matrix (3x3)
    - R: Rotation matrix (3x3)
    - t: Translation vector (3x1)

    Returns:
    - P: projection matrix (3x4)
    """
    Rt = np.concatenate((R, t), axis=1)
    P = K @ Rt
    return P

#distorsion function given the projection matrix and the distorsion coefficient
def distort(P, distCoeffs):
    P = Pi(P)
    r = np.linalg.norm(P, axis=0)
    dr = np.zeros(r.shape)
    power = 2
    for k in distCoeffs:
        dr += k * pow(r, power)
        power += 2

    return PiInv(P * (1 + dr))

def projected_points_with_P(P,Q):
    """
        Parameters:
    - P: 3x4 projection matrix
    - Q: 3xN matrix representing N 3D points in columns

    Returns:
    - 2xN projected point
    """
    q = P@PiInv(Q)
    return Pi(q)

#Projection points with distortion
def projectpoints(K:np.ndarray, R: np.ndarray, t: np.ndarray, Q: np.ndarray, distCoeffs=[], dist = False):
    """
        Parameters:
    - K: Camera matrix (3x3)
    - R: Rotation matrix (3x3)
    - t: Translation vector (3x1)
    - Q: 3xN matrix representing N 3D points in columns
    - distCoeffs: List of distortion coefficients ([])
    - dist: boolean to decide if the distortion is applied

    Returns:
    - 2xN matrix representing the projected 2D points in columns.
    """
    transformation_matrix = np.concatenate((R, t), axis=1)
    projection_matrix = transformation_matrix@PiInv(Q)
    if dist == True:
        distortion = distort(projection_matrix, distCoeffs)
        projection_points = K@distortion
    else:
        projection_points = K@projection_matrix
    return Pi(projection_points)

def undistortImage(im, K, distCoeffs):
    """
    Undistorts an image given camera matrix and distortion coefficients.

        Parameters:
            im: Input image (as a NumPy array).
            K: Camera intrinsic matrix (3x3).
            distCoeffs: List of distortion coefficients (1xn).

        Returns:
            Undistorted image.
    """
    # Create a grid of normalized pixel coordinates
    x, y = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
    p = np.stack((x, y, np.ones(x.shape))).reshape(3, -1)  # Homogeneous coordinates

    # Apply camera matrix to transform points to normalized image coordinates
    q = np.linalg.inv(K) @ p

    # Apply inverse distortion model
    q_d = distort(q, distCoeffs)
    p_d = K @ q_d

    # Reshape and convert to float32 for remapping
    x_d = p_d[0].reshape(x.shape).astype(np.float32)
    y_d = p_d[1].reshape(y.shape).astype(np.float32)

    # Sanity check (ensure homogeneous coordinates are valid)
    assert (p_d[2] == 1).all(), 'You did a mistake somewhere'

    # Apply remapping to get the undistorted image
    im_undistorted = cv2.remap(im, x_d, y_d, cv2.INTER_LINEAR)

    return im_undistorted

#Calculate Mapping from one point in one image plane to the other using a given H
def homography_mapping(q, H):
    ''' 
    Calculats mapping to image plane, given H matrix.

    Parameters:
    - q: 2D point in homogenous coordinate (3,N)
    - H: homography matrix

    Output:
    - q: 2D point in the corresponding image plane (2, N)
    '''
    return Pi(H@q)

def normalize2d(q):
    """
    Normalize 2D points.

    q : 2 x n, 2D points
    qn : 2 x n, normalized 2D points
    """
    if q.shape[0] != 2:
        raise ValueError("q must have 2 rows")
    if q.shape[1] < 2:
        raise ValueError("At least 2 points are required to normalize")

    mu = np.mean(q, axis=1).reshape(-1, 1)
    mu_x = mu[0].item()
    mu_y = mu[1].item()
    std = np.std(q, axis=1).reshape(-1, 1)
    std_x = std[0].item()
    std_y = std[1].item()
    Tinv = np.array([[std_x, 0, mu_x], [0, std_y, mu_y], [0, 0, 1]])
    T = np.linalg.inv(Tinv)
    qn = T @ PiInv(q)
    qn = Pi(qn)
    return qn, T

def hest(q1: np.ndarray,q2: np.ndarray, normalize=False):
    """
    Description:
    Calculates the B matrix by using kronecker multiplication and crossup setup.
    Estimates H given points q1, q2 on the two image planes. 
    Need at least 4 pairs of points to estimate the matrix with 8 degrees of freedom.
    Helper function to scale H to appropriate value
    Parameters:
    - q1 : 2 x n, 2D points in the first image
    - q2 : 2 x n, 2D points in the second image

    Output:
    - H: estimated homography matrix
    
    """
    if q1.shape[1] != q2.shape[1]:
        raise ValueError("Number of points in q1 and q2 must be equal")
    if q1.shape[1] < 4:
        raise ValueError(
            "At least 4 points are required to estimate a homography"
        )
    if q1.shape[0] != 2 or q2.shape[0] != 2:
        raise ValueError("q1 and q2 must have 2 rows")

    if normalize:
        q1, T1 = normalize2d(q1)
        q2, T2 = normalize2d(q2)

    n = q1.shape[1]
    B = []
    for i in range(n):
        x1, y1 = q1[:, i]
        x2, y2 = q2[:, i]
        Bi = np.array(
            [
                [0, -x2, x2 * y1, 0, -y2, y2 * y1, 0, -1, y1],
                [x2, 0, -x2 * x1, y2, 0, -y2 * x1, 1, 0, -x1],
                [-x2 * y1, x2 * x1, 0, -y2 * y1, y2 * x1, 0, -y1, x1, 0],
            ]
        )
        B.append(Bi)
    # with np.printoptions(precision=3, suppress=True):
    #     print("B:", B)
    B = np.array(B).reshape(-1, 9)
    U, S, Vt = np.linalg.svd(B)
    H = Vt[-1].reshape(3, 3)
    H = H.T
    if normalize:
        H = np.linalg.inv(T1) @ H @ T2
    return H

def crossOp(x: np.ndarray):
    """
    Takes a vector in 3D and returns the 3x3 matrix corresponding
    to taking the cross product with that vector.

    Parameters:
    - x: 1D vector

    Output:
    - 3x3 matrix
    """
    x = x.flatten()
    M = np.array([[0, -x[2], x[1]], 
                [x[2], 0, -x[0]],
                [-x[1], x[0], 0]])
    return M

def get_B(q1,q2):
    B = np.array([q1[0, 0]*q2[0, 0], q1[0, 0]*q2[1, 0], q1[0, 0],
                  q1[1, 0]*q2[0, 0], q1[1, 0]*q2[1, 0], q1[1, 0],
                  q2[0, 0], q2[1, 0], 1])
    for i in range(1,len(q1[0])):
        B_new_row = np.array([q1[0, i]*q2[0, i], q1[0, i]*q2[1, i], q1[0, i],
                              q1[1, i]*q2[0, i], q1[1, i]*q2[1, i], q1[1, i],
                              q2[0, i], q2[1, i], 1])
        B = np.vstack((B,B_new_row))
    return B

def fundamental_matrix(R1, t1, R2, t2, K1, K2):
    """
    The R and t relates the two cameras.
    K1 and K2 for each camera.
    Returns the F which relates the camera on the pixel plane.
    """
    R = R2@R1.T 
    t = t2 - R2@R1.T@t1 
    E = crossOp(t)@R
    F = (np.linalg.inv(K2)).T @ E @ (np.linalg.inv(K1))
    return F

def Fest_8point(q1, q2, normalize=True):
    ''' 
    Estimate F matrix from 8 points correspondences.
    It needs the helper function get_B

    Parameters:
    - q1,q2: 2D point in homogenous coordinate (3,N)

    Output:
    - F
    '''
    # Normalize the points
    if normalize:
        q1,T1 = normalize2d(q1)
        q2,T2 = normalize2d(q2)
    # Construct the B matrix from the matching points
    B = get_B(q1, q2)
    # Run the SVD algorithm: 0 = B@Flatten(F.T)
    _,_,VT = np.linalg.svd(B.T@B) #U,S,VT = np.linalg.svd(np.hstack((B.T,B)))
    # VT is sorted in descending order of singular value
    F_vec = VT[-1]
    F = np.array([[F_vec[0], F_vec[3], F_vec[6]],
                  [F_vec[1], F_vec[4], F_vec[7]],
                  [F_vec[2], F_vec[5], F_vec[8]]])
    if normalize:
        #return (np.linalg.inv(T1)@F.T@T2.round(2))
        return T2.T@F@T1
    else:
        return F
#Get 3D points from a list of 2D points and Projections Matrixes
def triangulate(q, P):

    """
    Return the triangulation.
    
    Parameters
    ----------
    q: 2 x n numpy array
        Inhomogenous pixel coordinates q1... qn
        One for each camera seeing the point.
        At least two.
    P: list of 3 x 4 numpy arrays
        Projection matrices P1... Pn
        For each pixel coordinate
    
    Return
    ------
    Q: 3 x 1 numpy array
        Triangulation of the point using the linear SVD algorithm

    Example:
        - q = np.hstack((q1_noise, q2_noise))
        - P = [P1,P2]
    """
    _, n = q.shape # n = no. cameras has seen pixel.

    # Prepare B matrix. Two rows for each camera n.
    B = np.zeros((2 * n, 4))
    for i in range(n):
        B[2 * i: 2 * i + 2] = [
            P[i][2, :] * q[0, i] - P[i][0, :],
            P[i][2, :] * q[1, i] - P[i][1, :],
        ]
    # BQ = 0. Minimize using Svd.
    _, _, vh = np.linalg.svd(B)
    Q = vh[-1, :] # Q is ev. corresponding to the min. singular point.
    return Q[:3].reshape(3, 1) / Q[3] # Reshape and scale.

def triangulate_nonlin(q, P):
    """
    Return the triangulation using nonlinear optimization.
    
    Parameters
    ----------
    q: 2 x n numpy array
        INHomogenous pixel coordinates q1... qn
        One for each camera seeing the point.
        At least two.
    P: list of 3 x 4 numpy arrays
        Projection matrices P1... Pn
        For each pixel coordinate
    
    Return
    ------
    Q: 3 x 1 numpy array
        Triangulation of the point using the linear SVD algorithm,
        combined with least square omptimizer
    Example:
        - q = np.hstack((q1_noise, q2_noise))
        - P = [P1,P2]
    """
    # Initial guess using SVD
    Q0 = triangulate(q, P)
    Q0 = Q0.reshape(3)
    
    def compute_residuals(Q):
        """
        In our case residuals is a vector of differences in the projections.
        Args:
            Q : 3-vector, 3D point (parameters to optimize)

        Returns:
            residuals : 2n-vector, residuals
                        (numbers to minimize sum of squares)
        """
        Qh = np.vstack((Q.reshape(3,1), 1))
        # n cameras
        n = len(q[0])
        residuals = np.zeros(shape=(2*n,))

        for i in range(n):
            qh_est = P[i] @ Qh
            q_est = qh_est[0:2, :]/qh_est[2, :]
            r = q_est - q[:,i].reshape(2, 1)

            residuals[2*i] = r[0]
            residuals[2*i+1] = r[1]


        return residuals
    Q = coolNonLinOptimizer.least_squares(compute_residuals, Q0)["x"].reshape(3,1)
    return  Q

def is_q_on_the_line(q,l,F):
    """""
    Verifing that a point is on the epipolar line

    Parameters: 
    - q: 2D point (2,1)
    - l: epipolar line in homogenous coordinate (3,1)
    - F: fundamental matrix

    Output:
    - It has to be almost 0
    """""
    #Verifing that q2 is located on the epipolar line
    is_q2_onTheLine = PiInv(q).T@F@PiInv(q)
    is_q2_onTheLine = PiInv(q).T @ l
    return print(is_q2_onTheLine)

def epipolar_line(F, q1):
    # returns epipolar line of q1 in camera 2
    return F@q1
#Given 3D points and 2D points, find the Projection Matrix
def pest(Q, q, normalize=False):
    """
    Estimate projection matrix using direct linear transformation.

    Args:
        Q : 3 x n array of 3D points
        q : 2 x n array of 2D points
        normalize : bool, whether to normalize the 2D points

    Returns:
        P : 3 x 4 projection matrix
    
    Example: 
        - Q = np.array([[0, 1, 0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]])
        - q = utilis.projectpoints(K, R, t, Q)

    """
    if Q.shape[0] != 3:
        raise ValueError("Q must be a 3 x n array of 3D points")
    if q.shape[0] != 2:
        raise ValueError("q must be a 2 x n array of 2D points")

    if normalize:
        q, T = normalize2d(q)

    q = PiInv(q)  # 3 x n
    Q = PiInv(Q)  # 4 x n
    n = Q.shape[1]  # number of points
    B = []
    for i in range(n):
        Qi = Q[:, i]
        qi = q[:, i]
        Bi = np.kron(Qi, crossOp(qi))
        B.append(Bi)
    B = np.array(B).reshape(3 * n, 12)
    U, S, Vt = np.linalg.svd(B)
    P = Vt[-1].reshape(4, 3)
    P = P.T
    if normalize:
        P = np.linalg.inv(T) @ P
    return P

# Compute the re-projection error
def compute_rmse(q_true, q_est):
    """
    Returns the root mean square error between the true and estimated 2D points.

    Args:
        q_true: 2 x n array of true 2D points
        q_est: 2 x n array of estimated 2D points
    """
    if q_true.shape[0] != 2 or q_est.shape[0] != 2:
        raise ValueError("q_true and q_est must be 2 in the first dimension")
    if q_true.shape[1] != q_est.shape[1]:
        raise ValueError("q_true and q_est must have the same number of points")
    se = (q_est - q_true) ** 2
    return np.sqrt(np.mean((se)))

def checkerboard_points(n: int,m: int):
    """""
    Return a list of 3D points

    Parameters:
        - n,m integers
    """""
    Q = np.zeros(shape=(3, n*m))
    for i in range(n):
        for j in range(m):
            Q[:, i*m+j] = [i - (n-1)/2.0,
                        j - (m-1)/2.0,
                        0]
                
    return Q
#1. Zhang method: finding a list of homographies given 3D points and 2D points
def estimate_homographies(Q_omega, qs):

    """
    Estimate homographies for each view.

    Args:
        Q_omega : 3 x (nxm) array of untransformed 3D points
        qs : list of arrays corresponding to each view (2, n),
             Same points, but rotated/scaled/transformed and projected to the image frame.

    Returns:
        Hs : list of 3x3 homographies for each view e.g (Hs[0],Hs[1]...Hs[len(qs)])

    In this case we can just assume that Q_omega is a virtual image plane with all the corners, and we find the homography mappings 
    from the image planes to this virtual image plane.

    """
    Hs = []
    Q = Q_omega[:2]  # remove 3rd row of zeros
    for q in qs:
        H = hest(q, Q)  # TODO: why hest(q, Q) instead of hest(Q, q)?
        Hs.append(H)
    return Hs

# 2. Zhang: function estimate_b(Hs) that takes a list of homographies Hs and returns the vector
def estimate_b(Hs):
    """
    Estimate b matrix used Zhang's method for camera calibration.

    Args:
        Hs : list of 3x3 homographies for each view

    Returns:
        b : 6x1 vector
    """
    V = []  # coefficient matrix
    # Create constraints in matrix form
    for H in Hs:
        vi_11 = form_vi(H, 1, 1)
        vi_12 = form_vi(H, 1, 2)
        vi_22 = form_vi(H, 2, 2)
        v = np.vstack((vi_12, vi_11 - vi_22))  # 2 x 6
        V.append(v)
    # V = np.array(V) creates the wrong array shape
    V = np.vstack(V)  # 2n x 6
    U, S, bt = np.linalg.svd(V.T @ V)
    b = bt[-1].reshape(6, 1)
    return b

# Helper function to form b
def form_vi(H, a, b):
    """
    Form 1x6 vector vi using H and indices alpha, beta.

    Args:
        H : 3x3 homography
        a, b : indices alpha, beta

    Returns:
        vi : 1x6 vector
    """
    # Use zero-indexing here. Notes uses 1-indexing.
    a = a - 1
    b = b - 1
    vi = np.array(
        [
            H[0, a] * H[0, b],
            H[0, a] * H[1, b] + H[1, a] * H[0, b],
            H[1, a] * H[1, b],
            H[2, a] * H[0, b] + H[0, a] * H[2, b],
            H[2, a] * H[1, b] + H[1, a] * H[2, b],
            H[2, a] * H[2, b],
        ]
    )
    vi = vi.reshape(1, 6)
    return vi

#Given B, find b
def b_from_B(B):
    """
    Returns the 6x1 vector b from the 3x3 matrix B.

    b = [B11 B12 B22 B13 B23 B33].T
    """
    if B.shape != (3, 3):
        raise ValueError("B must be a 3x3 matrix")

    b = np.array((B[0, 0], B[0, 1], B[1, 1], B[0, 2], B[1, 2], B[2, 2]))
    b = b.reshape(6, 1)
    return b

#3. Zhang: function estimateIntrisics(Hs) that takes a list of homographies Hs and returns a camera matrix K. 
def estimate_intrinsics(Hs):
    """
    Estimate intrinsic matrix using Zhang's method for camera calibration.

    Args:
        Hs : list of 3x3 homographies for each view

    Returns:
        K : 3x3 intrinsic matrix
    """
    b = estimate_b(Hs)
    B11, B12, B22, B13, B23, B33 = b
    # Appendix B of Zhang's paper
    v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)
    lambda_ = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11
    alpha = np.sqrt(lambda_ / B11)
    beta = np.sqrt(lambda_ * B11 / (B11 * B22 - B12**2))
    gamma = -B12 * alpha**2 * beta / lambda_
    u0 = lambda_ * v0 / beta - B13 * alpha**2 / lambda_
    # above values are sequences [value], so using [0] below is needed
    K = np.array([[alpha[0], gamma[0], u0[0]], [0, beta[0], v0[0]], [0, 0, 1]])
    return K

#4 Zhang: Rotations and translations
def estimate_extrinsics(K, Hs):
    """
    Estimate extrinsic parameters using Zhang's method for camera calibration.

    Args:
        K : 3x3 intrinsic matrix
        Hs : list of 3x3 homographies for each view

    Returns:
        Rs : list of 3x3 rotation matrices
        ts : list of 3x1 translation vectors
    """
    Kinv = np.linalg.inv(K)
    Rs = []
    ts = []
    for H in Hs:  # H = [h1|h2|h3]
        h1 = H[:, 0]
        h2 = H[:, 1]
        h3 = H[:, 2]
        lambda_ = np.linalg.norm(Kinv @ h1, 2)
        r1 = 1 / lambda_ * Kinv @ h1  # (3,)
        r2 = 1 / lambda_ * Kinv @ h2
        r3 = np.cross(r1, r2)
        t = np.array(1 / lambda_ * Kinv @ h3).reshape(3, 1)  # 3 x 1
        R = np.vstack((r1, r2, r3)).T  # 3 x 3 [r1|r2|r3]
        Rs.append(R)
        ts.append(t)
    Rs = np.array(Rs)
    ts = np.array(ts)
    return Rs, ts 

#5 Zhang: Calibrate Camera
# Compiling functions
def calibrate_camera(qs, Q):
    """
    Calibrate camera using Zhang's method for camera calibration.

    Args:
        qs : list of arrays corresponding to each view
        Q : 3 x (nxm) array of untransformed 3D points

    Returns:
        K : 3x3 intrinsic matrix
        Rs : list of 3x3 rotation matrices
        ts : list of 3x1 translation vectors

    Example: 
        - qs = [qa, qb, qc]
        - Q_omega = checkerboard_points(10, 20) (3,200)
    """
    Hs = estimate_homographies(Q, qs)
    K = estimate_intrinsics(Hs)
    Rs, ts = estimate_extrinsics(K, Hs)
    return K, Rs, ts

#######################################################################
#EX. 6 SMOOTHING, Corners detection and edge detection

def gaussian_1D_kernel(sigma, size=1):
    """
    Returns the 1D Gaussian kernel.

    Args:
        sigma: width of Gaussian kernel
        size (int): number of standard deviations to each side of the mean

    Returns:
        g: 1D Gaussian kernel
        gd: 1D Gaussian kernel derivative
    """
    if sigma == 0:
        return [1], [0]
    x = np.arange(-size * sigma, size * sigma + 1)
    g = gaussian(x,sigma)
    g = g / np.sum(g)  # normalize
    gd = gausDerivative(x,sigma)
    return g, gd

def gaussian(x,sigma):
    return np.exp(-x**2 / (2*sigma**2))

def gausDerivative(x,sigma):
    return -(x/sigma)*gaussian(x,sigma)

#Smoothing an Image
def gaussian_smoothing(im, sigma):
    """
    Smooths the input image with a 1D Gaussian kernel.

    Args:
        im : input image gray scale
        sigma : width of Gaussian kernel

    Returns:
        I : smoothed image
        Ix : image derivative in x-direction
        Iy : image derivative in y-direction

    Example: 
        im = cv2.imread(im, cv2.IMREAD_GRAYSCALE).astype(float)
    """
    g, gd = gaussian_1D_kernel(sigma)
    I = cv2.sepFilter2D(im, -1, g, g)
    Ix = cv2.sepFilter2D(im, -1, gd, g)
    Iy = cv2.sepFilter2D(im, -1, g, gd)
    return I, Ix, Iy

def structure_tensor(im, sigma, epsilon):
    I, Ix, Iy = gaussian_smoothing(im, sigma)
    g_eps, g_eps_d = gaussian_1D_kernel(epsilon)
    Ix_Iy = cv2.sepFilter2D(Ix * Iy, -1, g_eps, g_eps)
    Ix_x = cv2.sepFilter2D(Ix**2, -1, g_eps, g_eps)
    Iy_y = cv2.sepFilter2D(Iy**2, -1, g_eps, g_eps)
    C = np.array([[Ix_x, Ix_Iy],
                  [Ix_Iy, Iy_y]])
    return C

def harris_measure(im, sigma, epsilon, k):
    """
    Computes the Harris measure R(x,y) of the input image.

    Args:
        im : (h,w) input image
        sigma : Gaussian width to compute derivatives
        epsilon : Gaussian width to compute the structure tensor
        k : sensitivity factor

    Returns:
        r : (h,w), Harris measure
    """
    C = structure_tensor(im, sigma, epsilon)
    a = C[0, 0]
    b = C[1, 1]
    c = C[0, 1]
    r = a * b - c**2 - k * (a + b) ** 2
    return r

# Corner detection with non-max suppression
def corner_detector(im, sigma, epsilon, tau, k):
    """
    Detects corners in the input image using the Harris measure
    with non-max suprrssion and thresholding.

    Args:
        im : input image
        sigma : Gaussian width to compute derivatives
        epsilon : Gaussian width to compute the structure tensor
        tau : threshold for Harris measure
        k : sensitivity factor

    Returns:
        c : list of corner coordinates
    """
    r = harris_measure(im, sigma, epsilon, k)
    print(f"r: [{r.max():.2f}, {r.min():.2f}], tau = {tau/r.max():.2f}*r.max")

    # Perform 4-neigbourhood non-max suppression
    c = []
    for i in range(1, r.shape[0] - 1):
        for j in range(1, r.shape[1] - 1):
            if (
                r[i, j] > r[i + 1, j]
                and r[i, j] >= r[i - 1, j]
                and r[i, j] > r[i, j + 1]
                and r[i, j] >= r[i, j - 1]
                and r[i, j] > tau
            ):  # Threshold
                c.append([i, j])
    return c

def edge_detection(im, lower_thr, upper_thr):
    edges = cv2.Canny(im, lower_thr, upper_thr)
    return edges

############################################################################
#RANSAC

# Create a line from 2 points
def est_line(p1, p2):

    """
    Create a line from 2 points
    Args:
        - p1,p2: point (1,2)

    Returns:
        - l : line

    Example:
    """

    p1 = np.hstack((p1, 1))
    p2 = np.hstack((p2, 1))
    l = np.cross(p1, p2)
    return l

#determines which of a set of 2D points are an inliers or outliers with respect to a given line.
def is_inliner(point: np.ndarray, line: np.ndarray, threshold: int):
    """
    Determines which of a set of 2D points 
    are an inliers or outliers with respect to a given line.
    Args:
        point: point in homogenous coordinates (1,3)
        l : line (1,3)
        threshold: int 

    Returns:
        bool: True or False
        
    """
    line=line.copy()
    point=point.copy()
    # First we normalize a^2 + b^2 = 1
    a = line[0]
    b = line[1]
    scale = np.sqrt(a**2 + b**2)
    line/= scale

    # Then we can calculate distance to line:
    dist = np.abs(line.T@point)
    return dist<threshold

#Consensus
def consensus(line, points, threshold=1):
    '''
    Determines the number of inliers
    Parameters:
        - Line is a homogenous line.
        - Points is set of inhomogenous points.
    '''
    no = 0
    for i in range(len(points[0])):
        point = np.array([points[0, i],
                          points[1, i],
                          1])
        if is_inliner(point, line, threshold=threshold):
            no+=1
    return no

def drawTwo(points):
    
    '''
    Randomly samples two of n 2D points (without replacement).
    Parameters:
        - Points is set of inhomogenous points.
    '''
    indexes = np.random.choice(len(points[0]), 2)
    return points[:, indexes]

def pca_line(x): #assumes x is a (2 x n) array of points
    d = np.cov(x)[:, 0]
    d /= np.linalg.norm(d)
    l = [d[1], -d[0]]
    l.append(-(l@x.mean(1)))
    return l


def RANSAC_epsilon(points, threshold, p=0.99):
    best_inliers = 0
    best_line = None
    
    m = len(points[0])
    iteration = 0 # Current iteration
    N_hat = 1000 # Iterations needed to achieve p
    epsilon_hat = 1
    while iteration<N_hat:
        # Draw two random points
        p1, p2 = drawTwo(points)
        p1 = np.hstack((p1, 1))
        p2 = np.hstack((p2, 1))
        # Estimate the line
        line = est_line(p1, p2)
        # Find no. inliers
        no_inliers = consensus(line, points, threshold=threshold)
        if no_inliers>best_inliers:
            best_line = line.copy()
            best_inliers = no_inliers
            # Should we stop?
            epsilon_hat = 1 - best_inliers/m
            N_hat = np.log(1 - p) / np.log(1 - (1 - epsilon_hat)**2)
            print("eps_hat:", epsilon_hat, " ==> \tN-hat", N_hat, end="\n")
            print(f'Best line found: {best_line}')

        iteration += 1
    # We now have the best line. Extract inliners and fit.
    best_i = []
    for i in range(len(points[0])):
        p = np.hstack(        
            (points[:, i], 1))
        if is_inliner(p, best_line, threshold=threshold):
            best_i.append(i)
    best_points = points[:, best_i]
    
    fit_line = pca_line(best_points)
    print(f'Best line found: {best_line}') 
    print(f"no. of best_inliners: {best_points.shape[1]}")
    print(f'and num of iterations: {iteration}')
    return fit_line, best_points

def smallest_number_iterations(n_points: int, inliers: int, p: float, s: int ):
    '''
    Find the smallest number of iteration to fit 
    at least one model to only inliers
    Parameters:
        - n_points: Total number of points
        - inliers: Number of inliers
        - p: Desired probability
        - s: Fundamental Matrix (for stereo cameras): s=8,
            Homography: s=4 
            Essential Matrix s=5 

    Output: 
        N: number of iteration
    '''
    # Calculate epsilon (proportion of outliers)
    epsilon = 1 - (inliers / n_points)
    # Calculate N using the RANSAC formula for required iterations
    N = np.log(1 - p) / np.log(1 - (1 - epsilon)**s)
    return N


#################################################################
# BLOB DETECTION
# Read images as GRAY
def scale_spaced(im, sigma, n):
    """
    Naive implementation of the scale space pyramid with no downsampling.

    Args:
        im : input image img = img.astype(float).mean(2) / 255
        sigma : standard deviation of the Gaussian kernel
        n : number of scales

    Returns:
        im_scales : list containing the scale space pyramid of the input image
        scales : list containing the scales used in the pyramid
    """
    scales = [sigma * 2**i for i in range(n)]  # ratio = 2
    im_scales = []
    im_scale = im
    for scale in scales:
        # Apply Gaussian filter on the previously scaled image
        g, _ = gaussian_1D_kernel(scale)
        im_scale = cv2.sepFilter2D(src=im_scale, ddepth=-1, kernelX=g, kernelY=g)
        im_scales.append(im_scale)
    return im_scales, scales

def difference_of_gaussians(im, sigma, n):
    """
    Implementation of the difference of Gaussians.

    Args:
        im : input image
        sigma : standard deviation of the Gaussian kernel
        n : number of scales

    Returns:
        DoG : list of scale space DoGs of im
        scales : list containing the scales used in the pyramid
    """
    im_scales, scales = scale_spaced(im, sigma, n)
    DoG = []
    for i in range(1, n):
        diff = im_scales[i] - im_scales[i - 1]
        DoG.append(diff)
    return DoG, scales
# Ex 8.3
def detect_blobs(im, sigma, n, tau):
    """
    Implementation of the blob detector.

    Args:
        im : input image (in gray, float)
        sigma : standard deviation of the Gaussian kernel
        n : number of scales
        tau : threshold for blob detection

    Returns:
        blobs : list of detected blobs in the format (x, y, scale)
    """
    DoG, scales = difference_of_gaussians(im, sigma, n)
    DoG = np.array(DoG)

    # Obtain max value in a 3x3 neighborhood of each pixel in DoG
    MaxDoG = [cv2.dilate(abs(dog), np.ones((3, 3))) for dog in DoG] 

    # Thresholding & non-max suppression
    blobs = []
    prev_blobs = 0
    for i in range(len(DoG)):  # for each DoG

        if i==0:
            prev_MaxDoG = np.zeros(DoG[0].shape)
            next_MaxDoG = MaxDoG[i + 1]
        elif i==len(DoG)-1:
            prev_MaxDoG = MaxDoG[i - 1]
            next_MaxDoG = np.zeros(DoG[0].shape)
        else:
            prev_MaxDoG = MaxDoG[i - 1]
            next_MaxDoG = MaxDoG[i + 1]

        for j in range(im.shape[0]):  # for each row
            for k in range(im.shape[1]):  # for each column
                # take abs() to find max and min
                if (
                    abs(DoG[i][j, k]) > tau  # thresholding
                    and abs(DoG[i][j, k]) == MaxDoG[i][j, k] # max in current DoG
                    and abs(DoG[i][j, k]) > prev_MaxDoG[j, k] # max in previous DoG
                    and abs(DoG[i][j, k]) > next_MaxDoG[j, k] # max in next DoG
                ):
                    blobs.append((j, k, scales[i]))
        # Calculate how many new blobs detected in this DoG
        print(f"No. of blobs detected in DoG {i}: {len(blobs)-prev_blobs}")
        prev_blobs = len(blobs)
    return blobs

def visualize_blobs(blobs, im):
    """
    Args:
        blobs : list of detected blobs in the format (x, y, scale)
        im : BGR input image, not transformed in gray
    """
    gray_img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # To draw colored shapes on a gray img
    bgr_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)  
    for x, y, scale in blobs:
        cv2.circle(
            bgr_img, (y, x), radius=int(scale), color=(255, 0, 0), thickness=2
        )
    plt.axis("off")
    plt.imshow(bgr_img)
    plt.show()
#########################################################################
#SIFT

# Read images as uint8

def extract_coordinates_SIFT(image):
    """
    Args:
        - im : uint8 image 
    
    Output:
        - keypoint_coordinates: list of tuple
    
    Example:
        coordinates = extract_coordinates_sift(image)
        print("Coordinates of keypoints detected by SIFT:")
        for idx, coord in enumerate(coordinates):
            print(f"Keypoint {idx + 1}: (x, y) = ({coord[0]}, {coord[1]})")

    """

    # Check if the image is grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a SIFT detector object
    sift = cv2.SIFT_create()

    # Detect keypoints in the image
    keypoints = sift.detect(image, None)

    # Extract and return coordinates from the keypoints
    keypoint_coordinates = [kp.pt for kp in keypoints]  # list of tuples (x, y)
    return keypoint_coordinates

def SIFT_matcher(descriptors1, descriptors2, ratio_test=False):
    """
    Matches descriptors between two images using the Brute-Force Matcher.

    Parameters:
    - descriptors1: Descriptors from the first image.
    - descriptors2: Descriptors from the second image.
    - ratio_test: Boolean, whether to apply Lowe's ratio test to filter matches.

    Returns:
    - matches: A list of cv2.DMatch objects representing the matches.
    - index: store pairs of indices that represent the match connections 
            between keypoints from two images. A numpy array of tuples, each containing the indices (queryIdx, trainIdx) of matched descriptors
            from the first and second images respectively. This helps in identifying which descriptors (and
            therefore keypoints) from the first image correspond to which in the second image.
    """
    # Create a BFMatcher object with distance measurement cv2.NORM_L2 (Euclidean distance)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors.
    if ratio_test:
        # Find two nearest matches for each descriptor
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        # Apply Lowe's ratio test
        good_matches = []
        # m : Represents the best match for a given descriptor.
        # n : Represents the second-best match.
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        # m.queryIdx: The index of the descriptor in the first image's descriptor list that corresponds to this match.
        # m.trainIdx : The index of the descriptor in the second image's descriptor list that corresponds to this match
        index = np.array([(m.queryIdx, m.trainIdx) for m in good_matches])
        return good_matches, index
    else:
        # Find one best match for each descriptor
        matches = bf.match(descriptors1, descriptors2)
        # Sort them in the order of their distance (lowest distance first)
        matches = sorted(matches, key=lambda x: x.distance)
        index = np.array([(m.queryIdx, m.trainIdx) for m in matches])
        return matches, index
def extract_coordinates_and_best_match(matches, kp1, kp2, index, find_best_match=False):
    """
    Prints the coordinates of all matches and optionally finds and prints the best match.

    Parameters:
    - matches: A list of cv2.DMatch objects representing the matches.
    - kp1: Keypoints from the first image.
    - kp2: Keypoints from the second image.
    - index: A numpy array of tuples, each containing the indices (queryIdx, trainIdx) of matched descriptors.
    - find_best_match: Boolean, if True, finds and prints the coordinates of the best match based on the shortest distance.

    Outputs:
    - Prints coordinates of matches and optionally the best match.
    """
    if find_best_match:
        # Initialize variables to find the best match
        min_distance = float('inf')
        best_match = None
        best_coords = None

    # Iterate over all matches using the index array
    for idx, (queryIdx, trainIdx) in enumerate(index):
        # Get the coordinates from the keypoints
        (x1, y1) = kp1[queryIdx].pt
        (x2, y2) = kp2[trainIdx].pt
        print(f"Match {idx + 1}: Coordinates in image 1: ({x1}, {y1}), in image 2: ({x2}, {y2})")

        if find_best_match:
            # Check if the current match is the best match
            if matches[idx].distance < min_distance:
                min_distance = matches[idx].distance
                best_match = matches[idx]
                best_coords = ((x1, y1), (x2, y2))

    # If the best match flag is set, print the best match details
    if find_best_match and best_match is not None:
        print(f"Best match: Coordinates in image 1: {best_coords[0]}, in image 2: {best_coords[1]} with distance {best_match.distance}")   
#########################################################################################
#FUNDAMENTAL MATRIX ESTIMATION VIA RANSAC

###########First Method###############
def SIFT_extractor(gray_left, gray_right):
    """
    Extract matches, keypoints and points correspondances
    to matches

    Parameters:
    - 2 gray images

    Returns:
    - matches: A list of cv2.DMatch objects representing the matches.
    - np.array of the points correspondances
    - keypoints
    """
    # Use SIFT to getkeypoints and their descriptors
    sift = cv2.SIFT_create() # Sigma of the gaussian at octave 0
                                    
    kp1, des1 = sift.detectAndCompute(gray_left, None) #Keypoints and descriptors
    kp2, des2 = sift.detectAndCompute(gray_right, None) #Keypoints and descriptors

    # Match using brute force matcher, with crossCheck.
    bf = cv2.BFMatcher_create(crossCheck=True)
    matches = bf.match(des1, des2)

    p1s, p2s = [], []
    for match in matches:
        p1s.append(kp1[match.queryIdx].pt)
        p2s.append(kp2[match.trainIdx].pt)

    return matches, np.asarray(p1s).T, np.asarray(p2s).T, kp1, kp2

def SampsonsDistance(F, q1, q2):
    """
    params:
        - p1, p2 is 3x1 homogenous matching points
        - F is a 3x3 proposed fundamental matrix

    return: sampson distance between points
    """
    a1 = (q2.T@F)[0]**2
    b1 = (q2.T@F)[1]**2
    a2 = (F@q1)[0]**2
    b2 = (F@q1)[1]**2

    return (q2.T@F@q1)**2 * 1/(a1 + b1 + a2 + b2)

# Number of inliers given a threshold TODO 
def consensus_F(Fest, q1, q2, sigma: int):
    """
    Number of inliers for F estimation

    Parameters:
    - F: fundamental matrix
    - q1, q2: 2D point in homogenous coordinates (3, N)
    - sigma: 

    Returns:
    
    """
    threshold = 3.84*sigma**2
    inliers_index = []
    for i in range(len(q1[0])):
        sdist = SampsonsDistance(Fest, q1[:, i], q2[:, i])
        if sdist<threshold:
            inliers_index.append(i)
    return len(inliers_index), np.asarray(inliers_index)

def RANSAC_8_point(p1s, p2s, sigma=3, iterations=500):

    # RANSAC iterations
    best_inliers = 0
    Fbest = None
    for i in range(iterations):
        # Draw eight random matches
        match_idx = np.random.randint(low=0, high=len(p1s[0]), size=(8,))
        q1 = p1s[:, match_idx]
        q2 = p2s[:, match_idx]
        # Estimate the line
        Fest = Fest_8point(q1, q2, normalize=True)
        # Find no. inliers
        no_inliers, _ = consensus_F(Fest, p1s, p2s, sigma)
        if no_inliers>best_inliers:
            Fbest = Fest
            best_inliers = no_inliers
            print(i, "New best estimate!")
    print(f'Num of inliers: {best_inliers}')
    # Now, use all inliers of best F to match a final F
    _, inlier_index = consensus_F(Fbest, p1s, p2s, sigma)
    Final_F = Fest_8point(p1s[:, inlier_index], p2s[:, inlier_index], normalize=True)
    return Final_F, inlier_index, best_inliers
#########Second method################
def SampsonsDistance(F, q1, q2):
    return (q2.T@F@q1)**2 / ((q2.T@F)[0][0]**2 + (q2.T@F)[0][1]**2 + (F@q1)[0][0]**2 + (F@q1)[1][0]**2)

def get_inliers(matches, kps1, kps2, F, sigma):
    kps1 = [kps1[match.queryIdx].pt for match in matches]
    kps2 = [kps2[match.trainIdx].pt for match in matches]
    inliers = []
    min = math.inf
    for i in range(len(kps1)):
        kp1 = np.expand_dims(np.asarray(kps1[i]), axis=0).T
        kp2 = np.expand_dims(np.asarray(kps2[i]), axis=0).T
        err = SampsonsDistance(F, PiInv(kp1), PiInv(kp2))
        if err < min: min = err
        if err < 3.84*sigma**2: inliers.append(i)
    return inliers

def Fest_8point_A(q1, q2):
    B = []
    for i in range(q1.shape[1]):
        q1i = np.expand_dims(q1[:,i], axis=0)
        q2i = np.expand_dims(q2[:,i], axis=0)
        B.append((q1i.T @ q2i).flatten())
    B = np.vstack(B)
    U, S, VT = np.linalg.svd(B, full_matrices=True)
    F_flat = VT[-1]
    F = F_flat.reshape((3, 3)).T
    return F

def ransac(matches, kp1, kp2, N, sigma):
    best_inl = []
    for i in range(N):
        sample_kp1, sample_kp2 = sample(matches, kp1, kp2, size=8)
        F = Fest_8point_A(PiInv(sample_kp1), PiInv(sample_kp2))
        inliers = get_inliers(matches, kp1, kp2, F, sigma)
        if len(inliers) > len(best_inl):
            best_inl = inliers
    print(f'Num of inliers: {len(best_inl)}')
    return F, best_inl
##########################################################################################
#Homography MATRIX ESTIMATION VIA RANSAC
def sample(matches, kp1, kp2):
    indices = np.random.choice(len(matches), size=4, replace=False)
    sample_kp1 = [kp1[matches[i].queryIdx].pt for i in indices]
    sample_kp2 = [kp2[matches[i].trainIdx].pt for i in indices]
    return np.asarray(sample_kp1).T, np.asarray(sample_kp2).T

def get_inliers(matches, kps1, kps2, H, sigma):
    kps1 = [kps1[match.queryIdx].pt for match in matches]
    kps2 = [kps2[match.trainIdx].pt for match in matches]
    inliers = []
    min = math.inf
    for i in range(len(kps1)):
        kp1 = np.expand_dims(np.asarray(kps1[i]), axis=0).T
        kp2 = np.expand_dims(np.asarray(kps2[i]), axis=0).T
        d1 = np.square(kp1 - Pi(H @ PiInv(kp2))).sum()
        d2 = np.square(kp2 - Pi(np.linalg.inv(H) @ PiInv(kp1))).sum()
        err = math.sqrt(d1 + d2)
        if err < min: min = err
        if err < 5.99*sigma**2: inliers.append(i)
    return inliers

def ransac(matches, kp1, kp2, iterations, sigma):
    best_inl = []
    for i in range(iterations):
        sample_kp1, sample_kp2 = sample(matches, kp1, kp2)
        H = hest(sample_kp1, sample_kp2)
        inliers = get_inliers(matches, kp1, kp2, H, sigma)
        if len(inliers) > len(best_inl):
            best_inl = inliers
    print(f'Num of inliers: {len(best_inl)}')
    return best_inl

def estHomographyRANSAC(kp1, des1, kp2, des2):
    bf = cv2.BFMatcher_create(crossCheck=True)
    matches = bf.match(des1, des2)
    best_inl = ransac(matches, kp1, kp2, 200, 3)
    kps1 = [kp1[matches[i].queryIdx].pt for i in best_inl]
    kps2 = [kp2[matches[i].trainIdx].pt for i in best_inl]
    return hest(np.asarray(kps1).T, np.asarray(kps2).T)

##########################################################################
#Warp image
def warpImage(im, H, xRange, yRange):
    """
    Warp image, taken from exercise.

    Args:
        im (np.array): Image
        H (np.array): Homography matrix
        xRange (tuple): x-axis range for sampling
        yRange (tuple): y-axis range for sampling

    Returns:
        imWarp (np.array): Warped image, pixel array
        maskWarp (np.array): Warped mask, boolean array
    """
    T = np.eye(3)
    T[:2, 2] = [-xRange[0], -yRange[0]]
    H = T @ H
    outSize = (xRange[1] - xRange[0], yRange[1] - yRange[0])
    mask = np.ones(im.shape[:2], dtype=np.uint8) * 255
    imWarp = cv2.warpPerspective(im, H, outSize)
    maskWarp = cv2.warpPerspective(mask, H, outSize)
    return imWarp, maskWarp

########################################################################################
#Visual Odometry
def get_all_sift(gray, nfeatures=1999):
    """
    Parameters:
        - gray image
        - max number of SIFT features to extract

    Outuput:
        - point: 2d points extracted (N, 2)
        - descriptors, keypoints
    """
    sift = cv2.SIFT_create(nfeatures=nfeatures) 
                                
    kp, desc = sift.detectAndCompute(gray, None) #Keypoints and descriptors

    point = np.array([k.pt for k in kp]).astype(np.int32)
    
    return point, kp, desc

def match_sift(des1, des2, use_knn=False):
    """
    Matches SIFT descriptors between two images using Brute-Force matching.
    Optionally uses KNN matching with a ratio test for more robust matches.

    Parameters:
    - des1: Descriptors from the first image.
    - des2: Descriptors from the second image.
    - use_knn: Boolean flag to use KNN matching with Lowe's ratio test.

    Returns:
    - An array of tuples (queryIdx, trainIdx) of matched descriptors.
    """
    bf = cv2.BFMatcher()
    if use_knn:
        # Use KNN matcher and Lowe's ratio test
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    else:
        # Use simple brute force matcher
        matches = bf.match(des1, des2)
        good_matches = sorted(matches, key=lambda x: x.distance)
    
    # Extract and return indices of matches
    return np.array([(m.queryIdx, m.trainIdx) for m in good_matches])

def estimate_essential_matrix_and_pose(pt1, pt2, K, prob=0.999, threshold=1.0):
    """
    Estimates the essential matrix and decomposes it to find the correct camera pose.
    
    Parameters:
    - pt1: 2D points from the first image.
    - pt2: 2D points from the second image.
    - K: The camera intrinsic matrix.
    - prob: Probability of finding a good essential matrix.
    - threshold: Distance threshold for considering a point as an inlier.

    Returns:
    - E: The estimated essential matrix.
    - R: The rotation matrix.
    - t: The translation vector.
    - mask: Mask array of inliers that are in front of both cameras.
    """
    # Estimate the essential matrix using RANSAC
    E, mask = cv2.findEssentialMat(pt1, pt2, K, method=cv2.RANSAC, prob=prob, threshold=threshold)
    
    # Decompose the essential matrix to obtain the relative pose
    _, R, t, mask_pose = cv2.recoverPose(E, pt1, pt2, K, mask=mask)

    # Filter the mask to ensure both inliers from findEssentialMat and recoverPose are considered
    final_mask = mask.ravel().astype(bool) & mask_pose.ravel().astype(bool)
    print("Estimated Essential Matrix:\n", E)
    print("Rotation Matrix:\n", R)
    print("Translation Vector:\n", t)
    print("Number of Inliers in front of both cameras:", np.sum(mask))
    return E, R, t, final_mask

def estimate_pose_pnp(object_points, image_points, camera_matrix, dist_coeffs=None, use_extrinsic_guess=False):
    """
    Estimates the camera pose using the PnP (Perspective-n-Point) algorithm with RANSAC.

    Parameters:
    - object_points: 3D points in the world coordinate system (np.array of shape (N, 3)).
    - image_points: Corresponding 2D points on the image (np.array of shape (N, 2)).
    - camera_matrix: The camera intrinsic matrix (np.array of shape (3, 3)).
    - dist_coeffs: Lens distortion coefficients. None or np.array of shape (k,), default is zeros for no distortion.
    - use_extrinsic_guess: Boolean flag indicating if an initial guess of rotation and translation vectors is provided.

    Returns:
    - rvec: Rotation vector (np.array of shape (3, 1)).
    - tvec: Translation vector (np.array of shape (3, 1)).
    - inliers: Index of inlier points (np.array).
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    
    # Ensure data is in correct shape
    object_points = object_points.reshape(-1, 1, 3)
    image_points = image_points.reshape(-1, 1, 2)
    
    # Solve PnP with RANSAC
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points, image_points, camera_matrix, dist_coeffs,
        useExtrinsicGuess=use_extrinsic_guess, flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        raise Exception("solvePnPRansac did not converge!")
    print("Rotation Vector:\n", rvec)
    print("Translation Vector:\n", tvec)
    print("Inliers:\n", inliers)
    return rvec, tvec, inliers


def estimate_pose_from_matches(img_points1, img_points2, K, img_points3=None):
    """
    Estimates the camera pose using essential matrix and PnP methods.

    Parameters:
    - img_points1: Matched 2D points in the first image (np.array of shape (N, 2)).
    - img_points2: Corresponding matched 2D points in the second image (np.array of shape (N, 2)).
    - K: Camera intrinsic matrix (np.array of shape (3, 3)).
    - img_points3: Optional matched 2D points in the third image for further validation (np.array of shape (M, 2)).

    Returns:
    - E: Estimated essential matrix.
    - R: Rotation matrix from the essential matrix decomposition.
    - t: Translation vector from the essential matrix decomposition.
    - pose_inliers: Inliers from the pose estimation if a third image is provided.
    """
    # Estimate the essential matrix with RANSAC
    E, mask = cv2.findEssentialMat(img_points1, img_points2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # Decompose the essential matrix to obtain the pose
    _, R, t, pose_mask = cv2.recoverPose(E, img_points1, img_points2, K)

    if img_points3 is not None:
        # Assuming there's a third set of matches that need to be validated
        # Use intersect1d to find matches consistent across all images
        _, idx12, idx23 = np.intersect1d(pose_mask.ravel(), mask.ravel(), return_indices=True)
        valid_img_points1 = img_points1[idx12]
        valid_img_points3 = img_points3[idx23]

        # Refine pose estimation with PnP RANSAC using points from image 1 and image 3
        object_points = np.zeros((len(valid_img_points1), 3))  # Placeholder for 3D points
        _, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, valid_img_points3, K, None, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
        )
        pose_inliers = inliers
    else:
        pose_inliers = None

    return E, R, t, pose_inliers