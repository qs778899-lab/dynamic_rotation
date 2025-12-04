import numpy as np
import argparse
import os
from scipy.interpolate import CubicSpline


def read_pose_csv(csv_file):
    """
    Read 4x4 pose matrices from CSV file.
    Format: 4 rows per matrix, blank line between matrices.
    
    Returns:
        List of 4x4 numpy arrays
    """
    poses = []
    current_matrix = []
    
    with open(csv_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Non-empty line
                row = [float(x) for x in line.split(',')]
                current_matrix.append(row)
                if len(current_matrix) == 4:
                    poses.append(np.array(current_matrix))
                    current_matrix = []
            else:  # Empty line
                if current_matrix:
                    poses.append(np.array(current_matrix))
                    current_matrix = []
    
    # Handle last matrix if file doesn't end with blank line
    if current_matrix and len(current_matrix) == 4:
        poses.append(np.array(current_matrix))
    
    return poses


def write_pose_csv(csv_file, poses):
    """
    Write 4x4 pose matrices to CSV file in the same format as input.
    Format: 4 rows per matrix, blank line between matrices.
    """
    with open(csv_file, 'w') as f:
        for pose in poses:
            mat = np.array(pose, dtype=float).reshape(4, 4)
            for row in mat:
                f.write(','.join([f"{float(x):.6f}" for x in row]) + '\n')
            f.write('\n')


def select_keyframes(poses, start_idx, end_idx, num_keyframes):
    """
    Select keyframes from poses array.
    
    Args:
        poses: List of 4x4 pose matrices
        start_idx: Starting index
        end_idx: Ending index
        num_keyframes: Total number of keyframes to select (including start and end)
    
    Returns:
        selected_poses: List of selected pose matrices
        selected_indices: List of corresponding indices
    """
    total_poses = len(poses)
    start_idx = max(0, min(start_idx, total_poses - 1))
    end_idx = max(start_idx, min(end_idx, total_poses - 1))
    
    if num_keyframes <= 2:
        selected_indices = [start_idx, end_idx]
    else:
        # Uniformly sample keyframes between start and end
        selected_indices = np.linspace(start_idx, end_idx, num_keyframes, dtype=int).tolist()
    
    selected_poses = [poses[i] for i in selected_indices]
    
    return selected_poses, selected_indices


def interpolate_trajectory(keyframe_poses, keyframe_indices, num_output_frames):
    """
    Interpolate trajectory with smooth velocity profile.
    Start and end with low acceleration (slow), faster in the middle.
    
    Args:
        keyframe_poses: List of selected keyframe poses (4x4 matrices)
        keyframe_indices: Original indices of keyframes
        num_output_frames: Number of frames in output trajectory
    
    Returns:
        List of interpolated 4x4 pose matrices
    """
    if len(keyframe_poses) < 2:
        raise ValueError("Need at least 2 keyframes for interpolation")
    
    # Extract positions (translation vectors: 4th column, first 3 rows)
    keyframe_positions = np.array([pose[:3, 3] for pose in keyframe_poses])
    
    # Create time parameter for keyframes (normalize to [0, 1])
    keyframe_times = np.linspace(0, 1, len(keyframe_poses))
    
    # Create time parameter for output trajectory
    output_times = np.linspace(0, 1, num_output_frames)
    
    # Use cubic spline interpolation with zero velocity boundary conditions
    # bc_type='clamped' sets zero first derivative at boundaries (zero velocity)
    cs_x = CubicSpline(keyframe_times, keyframe_positions[:, 0], bc_type='clamped') #CubicSpline是三次样条插值
    cs_y = CubicSpline(keyframe_times, keyframe_positions[:, 1], bc_type='clamped')
    cs_z = CubicSpline(keyframe_times, keyframe_positions[:, 2], bc_type='clamped')
    
    # Interpolate positions
    interpolated_positions = np.column_stack([
        cs_x(output_times),
        cs_y(output_times),
        cs_z(output_times)
    ])
    
    # For rotation: use middle keyframe rotation as constant rotation
    # (Simple approach as requested - can be enhanced with SLERP later)
    middle_idx = len(keyframe_poses) // 2
    reference_rotation = keyframe_poses[middle_idx][:3, :3]
    
    # Create output poses
    interpolated_poses = []
    for pos in interpolated_positions:
        pose = np.eye(4)
        pose[:3, :3] = reference_rotation
        pose[:3, 3] = pos
        interpolated_poses.append(pose)
    
    return interpolated_poses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Interpolate object trajectory to generate smooth grasp trajectory'
    )
    
    parser.add_argument('--input_csv', type=str, default='RECORD/20251023_104936/pose_array.csv',
                        help='Input CSV file containing 4x4 pose matrices')
    parser.add_argument('--output_csv', type=str, default='interpolation_object_pose_trajectory.csv',
                        help='Output CSV file for interpolated trajectory')
    parser.add_argument('--start_index', type=int, default=5,
                        help='Starting index for keyframe selection')
    parser.add_argument('--end_index', type=int, default=-1,
                        help='Ending index for keyframe selection (-1 means last frame)')
    parser.add_argument('--num_keyframes', type=int, default=5,
                        help='Number of keyframes to select from input (including start and end)')
    parser.add_argument('--num_output_frames', type=int, default=600,
                        help='Number of frames in output interpolated trajectory')
    
    args = parser.parse_args()
    
    # Read input poses
    print(f"Reading poses from {args.input_csv}...")
    poses = read_pose_csv(args.input_csv)
    print(f"Loaded {len(poses)} poses from input file")
    
    if len(poses) == 0:
        raise ValueError("No poses found in input CSV file")
    
    # Handle end_index = -1 (use last frame)
    end_idx = args.end_index if args.end_index >= 0 else len(poses) - 1
    
    # Select keyframes
    print(f"\nSelecting {args.num_keyframes} keyframes from index {args.start_index} to {end_idx}...")
    keyframe_poses, keyframe_indices = select_keyframes(
        poses, args.start_index, end_idx, args.num_keyframes
    )
    print(f"Selected keyframes at indices: {keyframe_indices}")
    
    # Print keyframe positions
    print("\nKeyframe positions:")
    for i, (pose, idx) in enumerate(zip(keyframe_poses, keyframe_indices)):
        pos = pose[:3, 3]
        print(f"  Keyframe {i} (original index {idx}): [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
    
    # Interpolate trajectory
    print(f"\nInterpolating trajectory to {args.num_output_frames} frames...")
    print("Using cubic spline with clamped boundary conditions for smooth start/end")
    interpolated_poses = interpolate_trajectory(
        keyframe_poses, keyframe_indices, args.num_output_frames
    )
    
    # Determine output directory (same as input CSV directory if relative path used)
    if os.path.dirname(args.input_csv):
        output_dir = os.path.dirname(args.input_csv)
        output_path = os.path.join(output_dir, os.path.basename(args.output_csv))
    else:
        output_path = args.output_csv
    
    # Save interpolated trajectory
    print(f"\nSaving {len(interpolated_poses)} interpolated poses to {output_path}...")
    write_pose_csv(output_path, interpolated_poses)
    print("Done!")
    
    # Print some statistics
    print("\nTrajectory Statistics:")
    positions = np.array([pose[:3, 3] for pose in interpolated_poses])
    velocities = np.diff(positions, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    print(f"  Total distance: {np.sum(speeds):.4f} meters")
    print(f"  Max adjacent distance: {np.max(speeds):.6f} meters ({np.max(speeds)*1000:.3f} mm)")
    print(f"  Avg adjacent distance: {np.mean(speeds):.6f} meters ({np.mean(speeds)*1000:.3f} mm)")
    print(f"  Max speed: {np.max(speeds):.4f} m/frame")
    print(f"  Start speed: {speeds[0]:.4f} m/frame")
    print(f"  End speed: {speeds[-1]:.4f} m/frame")
    print(f"  Mid speed (avg of middle 10%): {np.mean(speeds[len(speeds)//2-5:len(speeds)//2+5]):.4f} m/frame")

