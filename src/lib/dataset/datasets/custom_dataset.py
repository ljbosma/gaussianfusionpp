from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ..generic_dataset import GenericDataset
from utils.ddd_utils import iou3d, compute_box_3d


def bbox_iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Both boxes must be in [x, y, w, h] format.
    """
    # Convert to [x1, y1, x2, y2]
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    x1_max = x1 + w1
    y1_max = y1 + h1
    x2_max = x2 + w2
    y2_max = y2 + h2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    iou = inter_area / union_area if union_area != 0 else 0
    return iou


def match_and_count(predictions, ground_truths, iou_threshold=0.5):
    """
    Matches predictions to ground truths and returns TP, FP, FN and match list.
    """
    matches = []
    tp, fp, fn = 0, 0, 0

    gt_by_image = {}
    for gt in ground_truths:
        gt_by_image.setdefault(gt['image_id'], []).append(gt)

    used_gt_ids = set()

    for pred in predictions:
        img_id = pred['image_id']
        matched = False
        best_iou = 0
        best_gt = None

        if img_id not in gt_by_image:
            fp += 1
            continue

        for gt in gt_by_image[img_id]:
            if gt['id'] in used_gt_ids:
                continue

            iou = bbox_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt = gt

        if best_gt:
            matches.append({
                "pred": pred,
                "ground_truth": best_gt
            })
            used_gt_ids.add(best_gt['id'])
            tp += 1
        else:
            fp += 1

    total_gt = sum(len(gts) for gts in gt_by_image.values())
    fn = total_gt - len(used_gt_ids)

    print("=============================================================================================================")
    print(f"Total number of predictions: {len(predictions)} | Ground truths: {len(ground_truths)}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}")

    return matches, tp, fp, fn


def accuracy_precision_recall_f1(tp, fp, fn):
  """Calculate accuracy, precision, recall, and F1 score based on predictions and ground truths."""

  accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
  precision = tp / (tp + fp) if (tp + fp) > 0 else 0
  recall = tp / (tp + fn) if (tp + fn) > 0 else 0
  f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

  return accuracy, precision, recall, f1

def average_cosine_distance(pred_angle, gt_angle):
  """Calculate the cosine distance for orientation."""
  # Convert angles from degrees to radians
  pred_rad = np.radians(pred_angle)
  gt_rad = np.radians(gt_angle)
  
  # Calculate vectors on the unit circle
  pred_vec = [np.cos(pred_rad), np.sin(pred_rad)]
  gt_vec = [np.cos(gt_rad), np.sin(gt_rad)]
  
  # Compute cosine of the angle between vectors
  cosine_similarity = np.dot(pred_vec, gt_vec)
  cosine_distance = 1 - cosine_similarity  # Since cosine similarity ranges from -1 to 1, distance ranges from 0 to 2
  return cosine_distance

def mean_squared_error(predictions, ground_truths, key):
  mse = np.mean([(pred[key] - gt[key]) ** 2 for pred, gt in zip(predictions, ground_truths)])
  return mse

def mean_absolute_error(predictions, ground_truths, key):
  mae = np.mean([abs(pred[key] - gt[key]) for pred, gt in zip(predictions, ground_truths)])
  return mae


def average_translation_error(pred_loc, gt_loc):
  """
  Calculate the Average Translation Error (ATE) in the xz plane.

  Parameters:
  pred_loc (np.array): Predicted location as [x, y, z].
  gt_loc (np.array): Ground truth location as [x, y, z].

  Returns:
  float: The 2D Euclidean distance between the predicted and ground truth locations in the xz plane.
  """
  # Extract x and z coordinates for both predicted and ground truth locations
  pred_xz = np.array([pred_loc[0], pred_loc[2]])
  gt_xz = np.array([gt_loc[0], gt_loc[2]])

  # Calculate the Euclidean distance in the xz plane
  return np.linalg.norm(pred_xz - gt_xz)


def iou3d_aligned(pred_dims, gt_dims):
  """
  Calculate the 3D IoU for aligned bounding boxes based on dimensions only.
  
  Parameters:
  pred_dims (np.array): Predicted dimensions as [width, height, length].
  gt_dims (np.array): Ground truth dimensions as [width, height, length].
  
  Returns:
  float: The IoU of the aligned bounding boxes.
  """
  intersection = np.prod(np.minimum(pred_dims, gt_dims))
  union = np.prod(pred_dims) + np.prod(gt_dims) - intersection
  return intersection / union

def average_scale_error(pred_dims, gt_dims):
  """
  Calculate the Average Scale Error (ASE).
  
  Parameters:
  pred_dims (np.array): Predicted dimensions as [width, height, length].
  gt_dims (np.array): Ground truth dimensions as [width, height, length].
  
  Returns:
  float: 1 minus the IoU of the aligned bounding boxes.
  """
  return 1 - iou3d_aligned(pred_dims, gt_dims)


def average_orientation_error(pred_yaw, gt_yaw):
  """
  Calculate the Average Orientation Error (AOE).
  
  Parameters:
  pred_yaw (float): Predicted yaw in radians.
  gt_yaw (float): Ground truth yaw in radians.
  
  Returns:
  float: The smallest yaw angle difference in radians.
  """
  return min(abs(pred_yaw - gt_yaw), 2 * np.pi - abs(pred_yaw - gt_yaw))

def average_velocity_error(pred_velocity, gt_velocity):
  """
  Calculate the Average Velocity Error (AVE).
  
  Parameters:
  pred_velocity (np.array): Predicted velocity as [vx, vz].
  gt_velocity (np.array): Ground truth velocity as [vx, vz].
  
  Returns:
  float: The Euclidean distance between the predicted and ground truth velocities.
  """
  pred_vel = np.array([pred_velocity[0], pred_velocity[2]])
  gt_vel = np.array([gt_velocity[0], gt_velocity[2]])
  return np.linalg.norm(pred_vel - gt_vel)


import numpy as np

def compute_nds(avg_ATE, avg_ASE, avg_AOE, avg_AVE, mAP=None):
    """
    Compute a normalized detection score (NDS) in [0, 1], based on weighted components:
    ATE, ASE, AOE, AVE, and optionally mAP.

    Normalization scheme:
      - ATE is divided by 100 (assumed upper bound of 100 meters)
      - ASE is already in [0, 1], no scaling needed
      - AOE is divided by π (maximum possible angular error)
      - AVE is unscaled (assumed to be already in m/s range)

    Parameters:
    - avg_ATE: Average Translation Error (in meters)
    - avg_ASE: Average Scale Error (1 - IoU)
    - avg_AOE: Average Orientation Error (in radians)
    - avg_AVE: Average Velocity Error (in m/s)
    - mAP: Optional mean Average Precision (range 0–1)

    Returns:
    - NDS: Normalized Detection Score ∈ [0, 1]
    """

    def safe_score(err): return max(1.0 - err, 0.0)

    ate_score = safe_score(avg_ATE / 100.0)
    ase_score = safe_score(avg_ASE)             # Already in [0, 1]
    aoe_score = safe_score(avg_AOE / np.pi)
    ave_score = safe_score(avg_AVE)             # Use raw value (assumes max reasonable value ~1)

    tp_scores = [ate_score, ase_score, aoe_score, ave_score]
    mAP_score = mAP if mAP is not None else 0.0

    nds = (4 * mAP_score + sum(tp_scores)) / 8.0
    return nds

def compute_metrics_from_matches(matches):
    ATEs, ASEs, AOE_s, AVE_s = [], [], [], []
    for match in matches:
        pred = match['pred']
        gt = match['ground_truth']
        ATEs.append(average_translation_error(np.array(pred['loc']), np.array(gt['location'])))
        ASEs.append(average_scale_error(np.array(pred['dim']), np.array(gt['dim'])))
        AOE_s.append(average_orientation_error(pred['rot_y'], gt['rotation_y']))
        AVE_s.append(average_velocity_error(np.array(pred['velocity']), np.array(gt['velocity_cam'][0:3])))

    avg_ATE = np.mean(ATEs) if ATEs else 0
    avg_ASE = np.mean(ASEs) if ASEs else 0
    avg_AOE = np.mean(AOE_s) if AOE_s else 0
    avg_AVE = np.mean(AVE_s) if AVE_s else 0

    return avg_ATE, avg_ASE, avg_AOE, avg_AVE



class CustomDataset(GenericDataset):
  num_categories = 1
  default_resolution = [-1, -1]
  class_name = ['obstacle']
  max_objs = 128
  cat_ids = {1: 1}

  def __init__(self, opt, split, eval_frustum=None):
    assert (opt.custom_dataset_img_path != '') and \
           (opt.custom_dataset_ann_path != '') and \
           (opt.num_classes != -1) and \
           (opt.input_h != -1) and (opt.input_w != -1), \
      'Missing custom dataset parameters: custom_dataset_img_path, custom_dataset_ann_path, num_classes, input_h, input_w.'

    img_dir = os.path.join(opt.custom_dataset_img_path, f"{split}")
    ann_path = os.path.join(opt.custom_dataset_ann_path, f"{split}.json")

    # Load annotations directly here
    with open(ann_path, 'r') as file:
        data = json.load(file)
    self.ground_truths = data['annotations']  # Now part of the instance

    self.num_categories = opt.num_classes
    self.class_name = ['obstacle'] if self.num_categories == 1 else [str(i) for i in range(self.num_categories)]
    self.default_resolution = [opt.input_h, opt.input_w]
    self.cat_ids = {i: i for i in range(1, self.num_categories + 1)}
    self.eval_frustum = eval_frustum

    self.images = None
    self.alpha_in_degree = False

    super().__init__(opt, split, ann_path, img_dir)
    self.num_samples = len(self.images)
    print(f"Loaded Custom dataset with {self.num_samples} samples for split: {split}")

  def __len__(self):
    return self.num_samples

  def _to_float(self, x):
    if isinstance(x, np.ndarray):
        x = x.item()
    return float("{:.2f}".format(x))

  def convert_eval_format(self, results):
    print("Converting evaluation format...")
    detections = []
    for image_id in results:
      for item in results[image_id]:
        if 'bbox' not in item or 'score' not in item or 'class' not in item:
          continue
        x1, y1, x2, y2 = item['bbox'][0:4]
        bbox = [x1, y1, x2 - x1, y2 - y1]
        detection = {
          "image_id": int(image_id),
          "category_id": int(item['class']),
          "bbox": list(map(self._to_float, bbox)),
          "score": self._to_float(item['score']),
          "dim": item['dim'].tolist(),
          "loc": item['loc'].tolist(),
          "velocity": item['velocity'].tolist(),
          "rot_y": self._to_float(item['rot_y']),
          "dep": self._to_float(item['dep']),
          "alpha": self._to_float(item['alpha'])
        }
        detections.append(detection)
    return detections

  
  def calculate_metrics_3d_iou(self, predictions, ground_truths):
    iou3d_scores = []
    iou2dbev_scores = []
    for pred, gt in zip(predictions, ground_truths):
      # Construct corners based on dim and loc
      pred_corners = compute_box_3d(np.array(pred['dim']), np.array(pred['loc']))
      gt_corners = compute_box_3d(np.array(gt['dim']), np.array(gt['loc']))
      
      iou3d_score, iou2d_score = iou3d(pred_corners, gt_corners)
      iou3d_scores.append(iou3d_score)
      iou2dbev_scores.append(iou2d_score)

    # Average IoU across all detections
    avg_iou3d = np.mean(iou3d_scores)
    avg_iou2dbev = np.mean(iou2dbev_scores)
    return avg_iou3d, avg_iou2dbev

  def save_results(self, results, save_dir):
    result_json = self.convert_eval_format(results)
    result_path = os.path.join(save_dir, f"results_custom_{self.split}.json")
    with open(result_path, 'w') as f:
      json.dump(result_json, f)
    print(f"Saved results to {result_path}")

  def run_eval(self, results, save_dir, n_plots, render_curves):
    print("Running custom evaluation...")

    # Convert predictions
    predictions = self.convert_eval_format(results)
    ground_truths = self.ground_truths

    matches, tp, fp, fn = match_and_count(predictions, ground_truths, iou_threshold=0.5)

    # Custom 3D/kinematic metrics
    ATEs, ASEs, AOE_s, AVE_s = [], [], [], []
    print(f"Number of matches = {len(matches)}")

    # Initialize grouped metric containers
    grouped_matches = {'0_100': [], '100_300': [], '300+': []}

    for match in matches:
        depth = match['ground_truth']['location'][2]
        if depth < 100:
            grouped_matches['0_100'].append(match)
        elif depth < 300:
            grouped_matches['100_300'].append(match)
        else:
            grouped_matches['300+'].append(match)

    # Global metrics
    avg_ATE, avg_ASE, avg_AOE, avg_AVE = compute_metrics_from_matches(matches)

    # Custom 2D detection stats
    accuracy, precision, recall, f1 = accuracy_precision_recall_f1(tp, fp, fn)

    # === COCO mean AP ===
    # Save prediction file
    result_json = self.convert_eval_format(results)
    result_path = os.path.join(save_dir, f"results_custom_{self.split}.json")
    with open(result_path, 'w') as f:
        json.dump(result_json, f)

    # Filter only what COCOeval expects
    coco_eval_json = []
    for det in self.convert_eval_format(results):
      coco_eval_json.append({
        "image_id": det["image_id"],
        "category_id": det["category_id"],
        "bbox": det["bbox"],
        "score": det["score"]
      })

    result_coco_path = os.path.join(save_dir, f"results_coco_{self.split}.json")

    with open(result_coco_path, 'w') as f:
      json.dump(coco_eval_json, f)

    # COCO Evaluation
    try:
        coco_gt_path = os.path.join(self.opt.custom_dataset_ann_path, f"coco_{self.split}.json")
        coco_gt = COCO(coco_gt_path)
        coco_dt = coco_gt.loadRes(result_coco_path)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        # Save COCO-style metrics
        coco_metrics = {
          "AP@[IoU=0.50:0.95]": coco_eval.stats[0],
          "AP@[IoU=0.50]": coco_eval.stats[1],
          "AP@[IoU=0.75]": coco_eval.stats[2],
          "AP_small": coco_eval.stats[3],
          "AP_medium": coco_eval.stats[4],
          "AP_large": coco_eval.stats[5],
          "AR_max1": coco_eval.stats[6],
          "AR_max10": coco_eval.stats[7],
          "AR_max100": coco_eval.stats[8],
          "AR_small": coco_eval.stats[9],
          "AR_medium": coco_eval.stats[10],
          "AR_large": coco_eval.stats[11],
        }
    except Exception as e:
        print("Failed to calculate COCO mAP:", e)
        coco_metrics = {
          "AP@[IoU=0.50:0.95]": 0,
          "AP@[IoU=0.50]": 0,
          "AP@[IoU=0.75]": 0,
          "AP_small": 0,
          "AP_medium": 0,
          "AP_large": 0,
          "AR_max1": 0,
          "AR_max10": 0,
          "AR_max100": 0,
          "AR_small": 0,
          "AR_medium": 0,
          "AR_large": 0,
        }

    # NDS
    nds = compute_nds(avg_ATE, avg_ASE, avg_AOE, avg_AVE, mAP=coco_metrics['AP@[IoU=0.50]'])

    metrics = {
        'Average Translation Error (ATE)': avg_ATE,
        'Average Scale Error (ASE)': avg_ASE,
        'Average Orientation Error (AOE)': avg_AOE,
        'Average Velocity Error (AVE)': avg_AVE,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'NDS': nds
    }
    metrics.update(coco_metrics)

    # Compute metrics for the different depth ranges
    depth_metrics = {}
    for key, match_list in grouped_matches.items():
      avg_ATE_g, avg_ASE_g, avg_AOE_g, avg_AVE_g = compute_metrics_from_matches(match_list)
      depth_metrics[f"metrics_{key}"] = {
          'Average Translation Error (ATE)': avg_ATE_g,
          'Average Scale Error (ASE)': avg_ASE_g,
          'Average Orientation Error (AOE)': avg_AOE_g,
          'Average Velocity Error (AVE)': avg_AVE_g,
          'Num Matches': len(match_list)
      }

    metrics.update(depth_metrics)

    out_dir = os.path.join(save_dir, 'custom_eval')
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, 'metrics_summary.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    print(f"Saved evaluation metrics to {metrics_path}")

    return out_dir


