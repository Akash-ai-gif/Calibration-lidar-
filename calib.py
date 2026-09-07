import numpy as np
import cv2 as cv
import glob
import os
import zipfile
import shutil
import sqlite3
import struct
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ──────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────

HD_DATA_DIR = r"C:\Users\skrak\Downloads\HD\extracted"
CALIBRATION_ZIP_DIR = r"C:\Users\skrak\Downloads\iiit dharwad dataset\calibration"
EXTRACTED_FRAMES_DIR = os.path.join(CALIBRATION_ZIP_DIR, "extracted_frames")

NROWS = 19
NCOLS = 12
SQUARE_SIZE = 0.022  # 2.2 cm

ORIENTATIONS = [
    'front horizontal', 'front vertical', 'left', 'right',
    'tilt left', 'tilt right'
]

# ──────────────────────────────────────────────────────────────────────
#  ROS2 Bag Frame Extraction Utilities
# ──────────────────────────────────────────────────────────────────────

def _align(offset, alignment=4):
    remainder = offset % alignment
    if remainder != 0:
        offset += alignment - remainder
    return offset

def _read_cdr_string(data, offset):
    length = struct.unpack_from('<I', data, offset)[0]
    offset += 4
    s = data[offset:offset + length - 1].decode('utf-8', errors='replace')
    offset += length
    offset = _align(offset)
    return s, offset

def _decode_ros2_image(data):
    offset = 4
    sec = struct.unpack_from('<i', data, offset)[0]; offset += 4
    nanosec = struct.unpack_from('<I', data, offset)[0]; offset += 4
    _frame_id, offset = _read_cdr_string(data, offset)
    height = struct.unpack_from('<I', data, offset)[0]; offset += 4
    width = struct.unpack_from('<I', data, offset)[0]; offset += 4
    encoding, offset = _read_cdr_string(data, offset)
    offset += 1
    offset = _align(offset)
    step = struct.unpack_from('<I', data, offset)[0]; offset += 4
    offset += 4
    img_bytes = data[offset:]
    return {
        'timestamp': sec + nanosec * 1e-9,
        'height': height, 'width': width,
        'encoding': encoding, 'step': step, 'data': img_bytes,
    }

def _image_to_bgr(info):
    h, w = info['height'], info['width']
    enc = info['encoding'].lower()
    raw = info['data']
    expected = h * w * 3
    if len(raw) < expected:
        buf = np.zeros(expected, dtype=np.uint8)
        buf[:len(raw)] = np.frombuffer(raw[:len(raw)], dtype=np.uint8)
    else:
        buf = np.frombuffer(raw[:expected], dtype=np.uint8)
    img = buf.reshape(h, w, 3)
    if enc == 'rgb8':
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    elif enc == 'bgra8':
        img = np.frombuffer(raw[:h * w * 4], dtype=np.uint8).reshape(h, w, 4)
        img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
    elif enc == 'rgba8':
        img = np.frombuffer(raw[:h * w * 4], dtype=np.uint8).reshape(h, w, 4)
        img = cv.cvtColor(img, cv.COLOR_RGBA2BGR)
    return img

def extract_frames_from_db3(db3_path, output_dir, prefix="frame"):
    os.makedirs(output_dir, exist_ok=True)
    conn = sqlite3.connect(db3_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, type FROM topics")
    topics = cursor.fetchall()
    image_topic_id = None
    for tid, name, msg_type in topics:
        if 'color/image_raw' in name and 'Image' in msg_type:
            image_topic_id = tid
            break
    if image_topic_id is None:
        conn.close()
        return 0
    cursor.execute(f"SELECT COUNT(*) FROM messages WHERE topic_id={image_topic_id}")
    total = cursor.fetchone()[0]
    cursor.execute(f"SELECT data, timestamp FROM messages WHERE topic_id={image_topic_id} ORDER BY timestamp")
    count = 0
    for row in cursor:
        raw_data, ts = row
        try:
            info = _decode_ros2_image(raw_data)
            img = _image_to_bgr(info)
            fname = f"{prefix}_{count:05d}.jpg"
            cv.imwrite(os.path.join(output_dir, fname), img)
            count += 1
        except Exception as e:
            count += 1
    conn.close()
    return count

def extract_all_calibration_frames(zip_dir=CALIBRATION_ZIP_DIR, output_dir=EXTRACTED_FRAMES_DIR):
    os.makedirs(output_dir, exist_ok=True)
    zip_files = glob.glob(os.path.join(zip_dir, '*.zip'))
    if not zip_files:
        print(f"No .zip files found in {zip_dir}")
        return
    total_frames = 0
    for zf in zip_files:
        basename = os.path.splitext(os.path.basename(zf))[0]
        prefix = basename.replace(' ', '_')
        existing = glob.glob(os.path.join(output_dir, f"{prefix}_*.jpg"))
        if existing:
            total_frames += len(existing)
            continue
        extract_to = os.path.join(zip_dir, f"_extracted_{basename}")
        try:
            if not os.path.exists(extract_to):
                with zipfile.ZipFile(zf, 'r') as z:
                    z.extractall(extract_to)
            db3_files = []
            for root, dirs, files in os.walk(extract_to):
                for f in files:
                    if f.endswith('.db3'):
                        db3_files.append(os.path.join(root, f))
            for db3 in db3_files:
                n = extract_frames_from_db3(db3, output_dir, prefix=prefix)
                total_frames += n
        finally:
            if os.path.exists(extract_to):
                shutil.rmtree(extract_to, ignore_errors=True)

# ──────────────────────────────────────────────────────────────────────
#  Camera Calibration (Intrinsics)
# ──────────────────────────────────────────────────────────────────────

def calibrate_from_hd_images():
    imgPathList = glob.glob(os.path.join(HD_DATA_DIR, '**', '*.jpeg'), recursive=True)
    if not imgPathList:
        print(f"No .jpeg images found in {HD_DATA_DIR}")
        return None, None

    print(f"\nCalibrating with {len(imgPathList)} HD images")
    print(f"Chessboard: {NROWS}x{NCOLS} inner corners, square={SQUARE_SIZE*1000:.0f}mm\n")

    worldPtsCur = np.zeros((NROWS * NCOLS, 3), np.float32)
    worldPtsCur[:, :2] = np.mgrid[0:NROWS, 0:NCOLS].T.reshape(-1, 2) * SQUARE_SIZE

    worldPtsList = []
    imgPtsList = []
    imgGray = None

    for idx, curImgPath in enumerate(imgPathList):
        imgBGR = cv.imread(curImgPath)
        if imgBGR is None:
            continue
        imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
        ok, corners = cv.findChessboardCornersSB(
            imgGray, (NROWS, NCOLS),
            cv.CALIB_CB_EXHAUSTIVE | cv.CALIB_CB_ACCURACY
        )
        if ok:
            worldPtsList.append(worldPtsCur)
            imgPtsList.append(corners)
            print(f"  [{idx+1}/{len(imgPathList)}] {os.path.basename(curImgPath)}: OK")
        else:
            print(f"  [{idx+1}/{len(imgPathList)}] {os.path.basename(curImgPath)}: FAILED")

    if len(imgPtsList) < 3:
        return None, None

    print("  Running calibrateCamera ...")
    repError, camMatrix, distCoeff, rvecs, tvecs = cv.calibrateCamera(
        worldPtsList, imgPtsList, imgGray.shape[::-1], None, None
    )
    print(f"\n  Camera Matrix:\n{camMatrix}")
    print(f"\n  Reprojection Error: {repError:.4f} pixels")

    curFolder = os.path.dirname(os.path.abspath(__file__))
    paramPath = os.path.join(curFolder, 'calibration_intrinsics.npz')
    np.savez(paramPath, repError=repError, camMatrix=camMatrix,
             distCoeff=distCoeff, rvecs=rvecs, tvecs=tvecs)
    return camMatrix, distCoeff

# ──────────────────────────────────────────────────────────────────────
#  PointCloud2 CDR Decoder
# ──────────────────────────────────────────────────────────────────────

def _decode_pointcloud2(data):
    offset = 4
    offset += 8
    fid_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    offset += fid_len
    if offset % 4 != 0: offset += 4 - (offset % 4)
    offset += 8
    n_fields = struct.unpack_from('<I', data, offset)[0]; offset += 4
    for _ in range(n_fields):
        name_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
        offset += name_len
        if offset % 4 != 0: offset += 4 - (offset % 4)
        offset += 4 + 1
        if offset % 4 != 0: offset += 4 - (offset % 4)
        offset += 4
    offset += 1
    if offset % 4 != 0: offset += 4 - (offset % 4)
    point_step = struct.unpack_from('<I', data, offset)[0]; offset += 4
    row_step = struct.unpack_from('<I', data, offset)[0]; offset += 4
    data_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    pc_data = data[offset:offset+data_len]
    dt = np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('intensity','<f4')])
    return np.frombuffer(pc_data, dtype=dt)

def extract_pointcloud_from_db3(db3_path, max_frames=9999):
    conn = sqlite3.connect(db3_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, type FROM topics")
    lidar_topic_id = None
    for tid, name, msg_type in cursor.fetchall():
        if 'PointCloud2' in msg_type:
            lidar_topic_id = tid
            break
    if lidar_topic_id is None:
        conn.close()
        return np.array([])
    cursor.execute(f"SELECT data FROM messages WHERE topic_id={lidar_topic_id} LIMIT {max_frames}")
    all_pts = []
    for row in cursor:
        pts = _decode_pointcloud2(row[0])
        valid = (pts['x'] != 0) | (pts['y'] != 0) | (pts['z'] != 0)
        all_pts.append(pts[valid])
    conn.close()
    return np.concatenate(all_pts) if all_pts else np.array([])

# ──────────────────────────────────────────────────────────────────────
#  LiDAR Board Plane Detection (RANSAC)
# ──────────────────────────────────────────────────────────────────────

def _ransac_plane(xyz, n_iter=3000, threshold=0.02):
    best_count = 0
    best_normal = None
    best_d = None
    best_mask = None
    n = len(xyz)
    for _ in range(n_iter):
        idx = np.random.choice(n, 3, replace=False)
        p1, p2, p3 = xyz[idx]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_mag = np.linalg.norm(normal)
        if norm_mag < 1e-10:
            continue
        normal /= norm_mag
        d = -np.dot(normal, p1)
        distances = np.abs(xyz @ normal + d)
        inliers = distances < threshold
        count = np.sum(inliers)
        if count > best_count:
            best_count = count
            best_normal = normal
            best_d = d
            best_mask = inliers
    return best_normal, best_d, best_mask

def find_board_in_lidar(pts):
    xyz = np.column_stack((pts['x'], pts['y'], pts['z']))
    intensity = pts['intensity']
    dist = np.linalg.norm(xyz, axis=1)
    mask = (dist > 0.5) & (dist < 12.0)
    xyz = xyz[mask]
    intensity = intensity[mask]
    if len(xyz) < 100:
        return None, None
    int_thresh = np.percentile(intensity, 75)
    hi_mask = intensity >= int_thresh
    xyz_hi = xyz[hi_mask]
    if len(xyz_hi) < 50:
        return None, None
    normal, d, inlier_mask = _ransac_plane(xyz_hi, n_iter=3000, threshold=0.02)
    if normal is None:
        return None, None
    inliers = xyz_hi[inlier_mask]
    centroid = np.mean(inliers, axis=0)
    centered = inliers - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[2]
    if np.dot(normal, -centroid) < 0:
        normal = -normal
    print(f"    RANSAC: {np.sum(inlier_mask)} inliers, centroid=({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")
    return centroid, normal

# ──────────────────────────────────────────────────────────────────────
#  REAL Extrinsic Calibration: LiDAR -> Camera
# ──────────────────────────────────────────────────────────────────────

def solve_extrinsics(camMatrix, distCoeff):
    print("\n" + "="*60)
    print("  REAL EXTRINSIC CALIBRATION (Target-Based)")
    print("="*60)

    board_pts_3d = np.zeros((NROWS * NCOLS, 3), np.float32)
    board_pts_3d[:, :2] = np.mgrid[0:NROWS, 0:NCOLS].T.reshape(-1, 2) * SQUARE_SIZE
    board_center_board = np.mean(board_pts_3d, axis=0)

    cam_normals = []
    cam_centers = []
    lidar_normals = []
    lidar_centers = []
    used_views = []

    for orient in ORIENTATIONS:
        folder = os.path.join(HD_DATA_DIR, orient)
        if not os.path.isdir(folder):
            continue
        db3_path = os.path.join(folder, 'my_scan_0.db3')
        jpegs = sorted(glob.glob(os.path.join(folder, '*.jpeg')))
        if not jpegs or not os.path.exists(db3_path):
            continue
        img_path = jpegs[0]
        print(f"\n  [{orient}]")
        
        img = cv.imread(img_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ok, corners = cv.findChessboardCornersSB(
            gray, (NROWS, NCOLS),
            cv.CALIB_CB_EXHAUSTIVE | cv.CALIB_CB_ACCURACY
        )
        if not ok:
            continue
        ok, rvec, tvec = cv.solvePnP(
            board_pts_3d, corners, camMatrix, distCoeff,
            flags=cv.SOLVEPNP_ITERATIVE
        )
        if not ok:
            continue
        R_board, _ = cv.Rodrigues(rvec)
        n_cam = R_board[:, 2].flatten()
        c_cam = (R_board @ board_center_board + tvec.flatten())
        if n_cam[2] < 0:
            n_cam = -n_cam

        pts = extract_pointcloud_from_db3(db3_path, max_frames=300)
        if len(pts) == 0:
            continue
        centroid_L, normal_L = find_board_in_lidar(pts)
        if centroid_L is None:
            continue

        cam_normals.append(n_cam)
        cam_centers.append(c_cam)
        lidar_normals.append(normal_L)
        lidar_centers.append(centroid_L)
        used_views.append(orient)

    n_views = len(used_views)
    print(f"\n  Paired {n_views} views: {used_views}")
    if n_views < 3:
        return np.eye(4, dtype=np.float64)

    N_cam = np.array(cam_normals)
    N_lidar = np.array(lidar_normals)
    H = N_lidar.T @ N_cam
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

    C_cam = np.array(cam_centers)
    C_lidar = np.array(lidar_centers)
    t = np.mean(C_cam - (R @ C_lidar.T).T, axis=0)

    def cost(params):
        rvec_opt = params[:3]
        t_opt = params[3:6]
        R_opt, _ = cv.Rodrigues(np.array(rvec_opt, dtype=np.float64))
        err = 0.0
        for i in range(n_views):
            n_pred = R_opt @ N_lidar[i]
            err += np.sum((n_pred - N_cam[i])**2) * 10.0
            c_pred = R_opt @ C_lidar[i] + t_opt
            err += np.sum((c_pred - C_cam[i])**2)
        return err

    rvec_init, _ = cv.Rodrigues(R)
    x0 = np.concatenate([rvec_init.flatten(), t])
    result = minimize(cost, x0, method='L-BFGS-B', options={'maxiter': 5000})

    R_final, _ = cv.Rodrigues(np.array(result.x[:3]))
    t_final = result.x[3:6]

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_final
    T[:3, 3] = t_final
    
    print(f"\n  Final T_cam_lidar:\n{T}")
    return T

# ──────────────────────────────────────────────────────────────────────
#  Projection & White-Square Filtering
# ──────────────────────────────────────────────────────────────────────

def build_white_square_mask(gray, corners):
    corner_grid = corners.reshape(NCOLS, NROWS, 2)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    n_white = 0
    for r in range(NCOLS - 1):
        for c in range(NROWS - 1):
            tl = corner_grid[r, c]
            tr = corner_grid[r, c + 1]
            bl = corner_grid[r + 1, c]
            br = corner_grid[r + 1, c + 1]
            quad = np.array([tl, tr, br, bl], dtype=np.int32)
            center = np.mean(quad, axis=0).astype(int)
            cx, cy = center[0], center[1]
            if 0 <= cx < gray.shape[1] and 0 <= cy < gray.shape[0]:
                if gray[cy, cx] > 128:
                    cv.fillConvexPoly(mask, quad, 255)
                    n_white += 1
    return mask, n_white

def project_and_filter(camMatrix, distCoeff, T_cam_lidar, img_path, db3_path):
    print(f"\n  Projecting: {os.path.basename(img_path)}")
    img = cv.imread(img_path)
    if img is None: return

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ok, corners = cv.findChessboardCornersSB(
        gray, (NROWS, NCOLS),
        cv.CALIB_CB_EXHAUSTIVE | cv.CALIB_CB_ACCURACY
    )
    if not ok: return

    mask, n_white = build_white_square_mask(gray, corners)
    
    # We only take high intensity points (which are actually the board)
    pts = extract_pointcloud_from_db3(db3_path)
    if len(pts) == 0: return
    
    # Actually use the intensity filter to ONLY project real white squares
    hi_mask = pts['intensity'] > np.percentile(pts['intensity'], 80)
    pts = pts[hi_mask]
    
    xyz = np.column_stack((pts['x'], pts['y'], pts['z'])).astype(np.float64)
    rvec, _ = cv.Rodrigues(T_cam_lidar[:3, :3])
    tvec = T_cam_lidar[:3, 3]
    proj, _ = cv.projectPoints(xyz, rvec, tvec, camMatrix, distCoeff)
    proj = proj.reshape(-1, 2)

    finite = np.all(np.isfinite(proj), axis=1)
    proj = proj[finite]
    ints = pts['intensity'][finite]

    valid = []
    for i, p in enumerate(proj):
        u, v = int(p[0]), int(p[1])
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            if mask[v, u] == 255:
                valid.append(i)

    final_pts = proj[valid]
    final_int = ints[valid]

    out_img = img.copy()
    for i, p in enumerate(final_pts):
        u, v = int(p[0]), int(p[1])
        t = min(final_int[i], 150) / 150.0
        color = (0, int(255 * (1 - t)), int(255 * t))
        cv.circle(out_img, (u, v), 4, color, -1)

    out_dir = os.path.join(HD_DATA_DIR, "projected_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "proj_" + os.path.basename(img_path))
    cv.imwrite(out_path, out_img)
    print(f"    Saved: {out_path}")

def runFullCalibration():
    print("\n" + "="*60)
    print("  STEP 1: Camera Intrinsic Calibration")
    print("="*60)
    camMatrix, distCoeff = calibrate_from_hd_images()
    if camMatrix is None: return

    T_cam_lidar = solve_extrinsics(camMatrix, distCoeff)

    print("\n" + "="*60)
    print("  STEP 3: Projecting LiDAR onto White Squares")
    print("="*60)
    for orient in ORIENTATIONS:
        folder = os.path.join(HD_DATA_DIR, orient)
        if not os.path.isdir(folder): continue
        db3_path = os.path.join(folder, 'my_scan_0.db3')
        jpegs = sorted(glob.glob(os.path.join(folder, '*.jpeg')))
        if not jpegs or not os.path.exists(db3_path): continue
        project_and_filter(camMatrix, distCoeff, T_cam_lidar, jpegs[0], db3_path)

if __name__ == '__main__':
    runFullCalibration()
