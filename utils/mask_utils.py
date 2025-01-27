import numpy as np

def calculate_cumulative_mask(pts3d, extrinsics, intrinsics):
    """
    Computes cumulative visibility masks for all views using their 3D points in the first frame's coordinates.
    
    Args:
        pts3d: 3D points in first frame's world coordinates (V, H, W, 3)
        extrinsics: Camera-to-world matrices (V, 4, 4)
        intrinsics: Camera intrinsics matrices (V, 3, 3)
        
    Returns:
        masks: Boolean array of shape (V, H, W) where True indicates novel pixels
    """
    V, H, W, _ = pts3d.shape
    masks = np.zeros((V, H, W), dtype=bool)

    # Precompute depth maps and world-to-camera transforms for all views
    depth_maps = []
    w2c_mats = []
    for i in range(V):
        # Calculate world-to-camera matrix
        c2w = extrinsics[i]
        w2c = np.linalg.inv(c2w)
        w2c_mats.append(w2c)
        
        # Transform points to camera space and extract depth
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        cam_pts = pts3d[i] @ R.T + t
        depth_maps.append(cam_pts[..., 2])  # (H, W)

    for i in range(V):
        # Current view's valid pixels
        valid_mask = depth_maps[i] > 1e-6
        
        if i == 0:
            masks[i] = valid_mask
            continue
            
        # Check visibility in previous views
        visible_in_prev = np.zeros_like(valid_mask)
        
        for j in range(i):
            # Get precomputed transforms for view j
            w2c_j = w2c_mats[j]
            K_j = intrinsics
            depth_j = depth_maps[j]
            
            # Transform current view's points to view j's camera space
            R_j = w2c_j[:3, :3]
            t_j = w2c_j[:3, 3]
            cam_pts_ij = pts3d[i] @ R_j.T + t_j  # (H, W, 3)
            
            # Project to view j's pixel coordinates
            x, y, z = np.split(cam_pts_ij, 3, axis=-1)  # Each becomes (H, W, 1)
            x = x.squeeze()
            y = y.squeeze()
            z = z.squeeze()
            
            # Avoid division by zero (invalid points will be masked later)
            valid_z = z > 1e-6
            z = np.where(valid_z, z, 1.0)  # Prevent NaNs
            
            # Get intrinsic parameters
            fx, fy = K_j[0, 0], K_j[1, 1]
            cx, cy = K_j[0, 2], K_j[1, 2]
            
            # Project to pixel coordinates
            u = (fx * x / z) + cx
            v = (fy * y / z) + cy
            
            # Create validity masks
            in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            valid_projection = valid_z & in_bounds

            # Convert coordinates to integer indices (nearest neighbor)
            u_idx = np.clip(np.round(u), 0, W-1).astype(int)
            v_idx = np.clip(np.round(v), 0, H-1).astype(int)
            
            # Get depth values from view j's depth map
            sampled_depth = np.zeros_like(z)
            sampled_depth[valid_projection] = depth_j[v_idx[valid_projection], 
                                                      u_idx[valid_projection]]
            
            # Depth consistency check (only on valid projections)
            depth_match = np.abs(z - sampled_depth) < 2e-1
            visible_in_j = valid_projection & depth_match
            
            visible_in_prev |= visible_in_j

        masks[i] = valid_mask & ~visible_in_prev

    return masks