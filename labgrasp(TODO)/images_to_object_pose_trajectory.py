import sys
sys.path.append("FoundationPose")
from estimater import *
from datareader import *
from dino_mask import get_mask_from_GD
import argparse
import os
import glob
import cv2
import imageio
import numpy as np


class RecordReader:
  def __init__(self, record_dir: str):
    self.record_dir = record_dir
    self.color_files = sorted(glob.glob(f"{record_dir}/color/*.png"))
    self.depth_files = sorted(glob.glob(f"{record_dir}/depth/*.png"))

  def get_color(self, i: int):
    return imageio.imread(self.color_files[i])

  def get_depth(self, i: int):
    # depth stored as uint16 in mm; convert to float meters
    depth_mm = cv2.imread(self.depth_files[i], -1)
    if depth_mm is None:
      return None
    return depth_mm.astype(np.float32) / 1000.0


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument('--record_dir', type=str, default='/home/erlin/work/labgrasp/RECORD/20251023_104936',
                      help='Directory containing color/ and depth/ folders')
  parser.add_argument('--mesh_file', type=str, default='mesh/1cm_10cm.obj')
  parser.add_argument('--cam_K_file', type=str, default='cam_K.txt')
  parser.add_argument('--est_refine_iter', type=int, default=15)
  parser.add_argument('--step', type=int, default=1,
                      help='Compute pose every N frames')
  parser.add_argument('--start_index', type=int, default=10)
  parser.add_argument('--end_index', type=int, default=115)
  parser.add_argument('--mask_prompt', type=str, default='red cylinder')
  parser.add_argument('--display', action='store_true', default=False)
  parser.add_argument('--block_display', action='store_true', default=False,
                      help='Use waitKey(0) and block per frame')
  parser.add_argument('--save_track_vis', action='store_true', default=False,
                      help='Save visualization images to debug_dir/track_vis')
  parser.add_argument('--output_csv', type=str, default='pose_array.csv')
  parser.add_argument('--debug_dir', type=str, default='debug')
  parser.add_argument('--debug', type=int, default=0)

  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  # Load mesh and compute oriented bounds (same behavior as run_demo_record)
  mesh = trimesh.load(args.mesh_file)
  mesh.vertices /= 1000.0
  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  # Estimator init (mirrors grasp_main.py lines ~76-82)
  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                       scorer=scorer, refiner=refiner, debug_dir=args.debug_dir, debug=args.debug, glctx=glctx)
  logging.info("estimator initialization done")

 
  cam_k = np.loadtxt(args.cam_K_file).reshape(3,3)

  # Prepare debug directories
  os.system(f'rm -rf {args.debug_dir}/* && mkdir -p {args.debug_dir}/track_vis {args.debug_dir}/ob_in_cam')

 
  reader = RecordReader(args.record_dir)
  num_frames = len(reader.color_files)
  if num_frames == 0:
    raise RuntimeError(f"No color frames found in {os.path.join(args.record_dir, 'color')}")

  start_idx = max(0, args.start_index)
  end_idx = min(args.end_index, num_frames - 1)

  pose_array = []
  frame_indices = []

  for i in range(num_frames):
    logging.info(f'i:{i}')
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    if depth is None:
      continue

    if (i % args.step == 0) and (i >= start_idx) and (i <= end_idx):
      # Generate mask using GroundingDINO
      mask = get_mask_from_GD(color, args.mask_prompt)

      # FoundationPose registration
      pose = est.register(K=cam_k, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

      # Convert to object center pose
      center_pose = pose @ np.linalg.inv(to_origin)

      print(f"第{i}帧检测完成，center pose of object: {center_pose}")

      # Visualization
      if args.display or args.save_track_vis:
        vis = draw_posed_3d_box(cam_k, img=color, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=cam_k, thickness=3, transparency=0, is_input_rgb=True)
        if args.display:
          cv2.imshow('pose_vis', vis[...,::-1])
          if args.block_display:
            cv2.waitKey(0)
          else:
            cv2.waitKey(1)
        if args.save_track_vis:
          os.makedirs(os.path.join(args.debug_dir, 'track_vis'), exist_ok=True)
          out_path = os.path.join(args.debug_dir, 'track_vis', f'frame_{i:06d}.png')
          imageio.imwrite(out_path, vis)

      pose_array.append(np.array(center_pose, dtype=float))
      frame_indices.append(i)
      logging.info(f'Pose computed for frame {i}')

    #   cv2.waitKey(0) #waitKey(0) 是一种阻塞
    #   input("break001") #input也是一种阻塞

  # Save poses to CSV ，preserving 4x4 structure (4 lines per pose, blank line between poses)
  if len(pose_array) > 0:
    output_csv_file = os.path.join(args.record_dir, args.output_csv)  #weight save path
    with open(output_csv_file, 'w') as f:
      for pose in pose_array:
        mat = np.array(pose, dtype=float).reshape(4,4)
        for row in mat:
          f.write(','.join([f"{float(x):.6f}" for x in row]) + '\n')
        f.write('\n')
    logging.info(f'Saved {len(pose_array)} poses to {output_csv_file}')

    # Also save frame indices like run_demo_record (optional utility file)
    frame_info_file = os.path.join(args.record_dir, 'pose_frame_indices.txt')
    np.savetxt(frame_info_file, np.array(frame_indices, dtype=int), fmt='%d')
    logging.info(f'Saved frame indices to {frame_info_file}')
  else:
    logging.info('No poses computed - nothing to save')

  if args.display:
    cv2.destroyAllWindows()


