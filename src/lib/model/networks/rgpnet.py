import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import _nms, _sigmoid, _topk, _tranpose_and_gather_feat
from utils.ddd_utils import alpha2rot_y_torch, unproject_2d_to_3d_torch, project_to_image_torch
from utils.image import transform_preds_with_trans_torch
from utils.pointcloud import get_dist_thresh_torch, get_alpha
import numpy as np

class RadarGaussianParamNet(nn.Module):
    def __init__(self, opt, input_dim=2):
        super(RadarGaussianParamNet, self).__init__()

        self.hidden_dim = opt.rgp_hidden_dim
        self.num_layers = opt.rgp_num_layers

        layers = []
        in_dim = input_dim

        for i in range(self.num_layers):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            if opt.rgp_bn_hidden_layers:
                layers.append(nn.BatchNorm1d(self.hidden_dim))
            if opt.rgp_dropout:
                layers.append(nn.Dropout(p=opt.rgp_dropout_probability))
            layers.append(nn.ReLU())
            in_dim = self.hidden_dim  # next input dim

        self.mlp = nn.Sequential(*layers)

        self.mean_head = nn.Linear(self.hidden_dim, 2)  # [μ_x, μ_d]
        self.cov_head = nn.Linear(self.hidden_dim, 3)   # [log_σ_x, log_σ_d, tanh_ρ]

    def forward(self, radar_points):
        B, N, D = radar_points.shape
        x = radar_points.view(B * N, D)
        feats = self.mlp(x).view(B, N, -1)
        pooled = feats.max(dim=1)[0]

        mean = self.mean_head(pooled)
        cov_params = self.cov_head(pooled)

        log_std = cov_params[:, :2]
        log_std = torch.clamp(log_std, min=-5.0, max=5.0)
        rho = torch.clamp(torch.tanh(cov_params[:, 2]), min=-0.99, max=0.99)

        std = torch.exp(log_std)
        cov_matrices = []
        for i in range(B):
            sx, sd = std[i]
            r = rho[i]
            cov = torch.tensor([
                [sx ** 2, r * sx * sd],
                [r * sx * sd, sd ** 2]
            ], device=radar_points.device)
            cov_matrices.append(cov)

        cov_matrices = torch.stack(cov_matrices)
        return mean, cov_matrices


    def generate_rgp_pc_box_hm_torch(self, output, pc_hm, calibs, opt, trans_originals):
        K = opt.max_objs
        phase = 'val'
        device = pc_hm.device
        batch_size = pc_hm.size(0)

        heat = _nms(_sigmoid(output['hm'].clone()))
        scores, inds, clses, ys0, xs0 = _topk(heat, K=K)
        scores = scores.view(batch_size, K, 1, 1)
        xs = xs0.view(batch_size, K, 1) + 0.5
        ys = ys0.view(batch_size, K, 1) + 0.5

        wh = _tranpose_and_gather_feat(output['wh'], inds).view(batch_size, K, -1)
        wh[wh < 0] = 0
        bboxes = torch.cat([xs - wh[..., 0:1] / 2, ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2, ys + wh[..., 1:2] / 2], dim=2)

        out_dep = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
        dep = _tranpose_and_gather_feat(out_dep, inds)
        dims = _tranpose_and_gather_feat(output['dim'], inds).view(batch_size, K, -1)
        rot = _tranpose_and_gather_feat(output['rot'], inds).view(batch_size, K, -1)

        pc_box_hm = torch.zeros_like(pc_hm)

        for i, [pc_hm_b, bboxes_b, depth_b, dim_b, rot_b, score_b] in enumerate(zip(pc_hm, bboxes, dep, dims, rot, scores)):
            alpha_b = get_alpha(rot_b).unsqueeze(1)
            calib_b = calibs[i]
            trans_original_b = trans_originals[i]

            if opt.sort_det_by_depth:
                idx = torch.argsort(depth_b[:, 0])
                bboxes_b = bboxes_b[idx]
                depth_b = depth_b[idx]
                dim_b = dim_b[idx]
                rot_b = rot_b[idx]
                alpha_b = alpha_b[idx]
                score_b = score_b[idx]

            for j, [bbox, depth, dim, alpha, score] in enumerate(zip(bboxes_b, depth_b, dim_b, alpha_b, score_b)):
                if score < opt.rgp_pred_thresh:
                    continue

                ct = torch.tensor([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], device=device)
                ct_trans = transform_preds_with_trans_torch(ct, trans_original_b.clone())
                location = unproject_2d_to_3d_torch(ct_trans, depth, calib_b)
                dist_thresh = get_dist_thresh_torch(calib_b, ct_trans, dim, alpha, opt, phase=phase, location=location, device=device)

                w = bbox[2] - bbox[0]
                expand_pixels = w * (opt.frustum_expand_x + opt.dynamicFrustumExpansionRatio * depth.item()**2)
                bbox = bbox.clone()  # Make a new tensor to avoid in-place ops on a view
                bbox[0] = bbox[0] - expand_pixels / 2
                bbox[2] = bbox[2] + expand_pixels / 2


                bbox_int = torch.tensor([torch.floor(bbox[0]), torch.floor(bbox[1]),
                                        torch.ceil(bbox[2]), torch.ceil(bbox[3])], dtype=torch.int32)

                roi = pc_hm_b[:, bbox_int[1]:bbox_int[3]+1, bbox_int[0]:bbox_int[2]+1]
                pc_dep = roi[opt.pc_feat_channels['pc_dep']]
                nonzero_inds = torch.nonzero(pc_dep > 0, as_tuple=True)

                if len(nonzero_inds[0]) == 0:
                    continue

                xs_roi = nonzero_inds[1].float() + bbox_int[0].float()
                depths_roi = pc_dep[nonzero_inds[0], nonzero_inds[1]]
                coords = torch.stack([xs_roi, depths_roi], dim=1)

                if opt.use_dist_for_frustum:
                    pc_pos_x = roi[-2]
                    pc_pos_z = roi[-1]
                    pc_dist = torch.sqrt(torch.square(pc_pos_x) + torch.square(pc_pos_z))
                    pc_dist = pc_dist[nonzero_inds[0], nonzero_inds[1]]
                    within_thresh = (pc_dist < dist_thresh[1]) & (pc_dist > dist_thresh[0])
                else:
                    within_thresh = (depths_roi < depth + dist_thresh) & (depths_roi > max(0, depth - dist_thresh))

                coords = coords[within_thresh].unsqueeze(0)

                if coords.shape[1] == 0:
                    continue

                mean, cov = self.forward(coords)
                diff = coords[0] - mean[0]
                inv_cov = torch.inverse(cov[0])
                weights = torch.exp(-0.5 * (diff @ inv_cov * diff).sum(dim=1))
                weights_sum = weights.sum()
                if weights_sum < 1e-8:
                    weights_sum = torch.tensor(1e-8, device=weights.device)
                weights = weights / weights_sum
                weights = weights / (weights.sum() + 1e-6)

                weighted_depth = (weights * coords[0][:, 1]).sum()
                if opt.normalize_depth:
                    weighted_depth /= opt.max_pc_depth

                w_interval = opt.hm_to_box_ratio * (bbox[2] - bbox[0])
                h_interval = opt.hm_to_box_ratio * (bbox[3] - bbox[1])
                w_min = int(ct[0] - w_interval / 2.)
                w_max = int(ct[0] + w_interval / 2.)
                h_min = int(ct[1] - h_interval / 2.)
                h_max = int(ct[1] + h_interval / 2.)

                H, W = pc_box_hm.shape[2], pc_box_hm.shape[3]
                h_min = max(0, h_min)
                h_max = min(H - 1, h_max)
                w_min = max(0, w_min)
                w_max = min(W - 1, w_max)

                pc_box_hm[i, opt.pc_feat_channels['pc_dep'], h_min:h_max+1, w_min:w_max+1] = weighted_depth

        return pc_box_hm
    

    def generate_rgp_pc_box_hm_torch_with_ann(self, output, pc_hm, calibs, opt, trans_originals, batch):
        K = opt.max_objs
        phase = 'val'
        device = pc_hm.device
        batch_size = pc_hm.size(0)
        if batch['reg_mask'].sum() == 0:
            # No ground-truth objects in the batch
            pc_box_hm = torch.zeros_like(pc_hm)
            dummy_mean = torch.zeros(1, opt.max_objs, 2, device=pc_hm.device)
            dummy_cov = torch.zeros(1, opt.max_objs, 2, 2, device=pc_hm.device)
            dummy_valid_mask = torch.zeros(1, opt.max_objs, dtype=torch.bool, device=pc_hm.device)
            dummy_gt_coords = torch.full((1, opt.max_objs), -1, device=pc_hm.device, dtype=torch.int64)
            return pc_box_hm, dummy_mean, dummy_cov, dummy_valid_mask, dummy_gt_coords


        heat = _nms(_sigmoid(output['hm'].clone()))
        scores, inds, clses, ys0, xs0 = _topk(heat, K=K)
        scores = scores.view(batch_size, K, 1, 1)
        xs = xs0.view(batch_size, K, 1) + 0.5
        ys = ys0.view(batch_size, K, 1) + 0.5

        wh = _tranpose_and_gather_feat(output['wh'], inds).view(batch_size, K, -1)
        wh[wh < 0] = 0
        bboxes = torch.cat([xs - wh[..., 0:1] / 2, ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2, ys + wh[..., 1:2] / 2], dim=2)

        out_dep = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
        dep = _tranpose_and_gather_feat(out_dep, inds)
        dims = _tranpose_and_gather_feat(output['dim'], inds).view(batch_size, K, -1)
        rot = _tranpose_and_gather_feat(output['rot'], inds).view(batch_size, K, -1)

        pc_box_hm = torch.zeros_like(pc_hm)
        all_means = []
        all_covs = []
        all_valids = []
        matched_gt_coords = []

        for i, [pc_hm_b, bboxes_b, depth_b, dim_b, rot_b, score_b] in enumerate(zip(pc_hm, bboxes, dep, dims, rot, scores)):
            alpha_b = get_alpha(rot_b).unsqueeze(1)
            calib_b = calibs[i]
            trans_original_b = trans_originals[i]

            if opt.sort_det_by_depth:
                idx = torch.argsort(depth_b[:, 0])
                bboxes_b = bboxes_b[idx]
                depth_b = depth_b[idx]
                dim_b = dim_b[idx]
                rot_b = rot_b[idx]
                alpha_b = alpha_b[idx]
                score_b = score_b[idx]

            valid_mask = batch['reg_mask'][i] > 0
            gt_locs = batch['location'][i][valid_mask.squeeze(-1)]  # [N, 3]
            proj_gt = project_to_image_torch(gt_locs.T, calib_b)  # [N, 2]
            gt_xs = proj_gt[0, :]
            gt_zs = gt_locs[:, 2]

            for j, [bbox, depth, dim, alpha, score] in enumerate(zip(bboxes_b, depth_b, dim_b, alpha_b, score_b)):
                if score < opt.rgp_pred_thresh:
                    all_means.append(torch.zeros(1, 2, device=device))
                    all_covs.append(torch.zeros(1, 2, 2, device=device))
                    all_valids.append(False)
                    matched_gt_coords.append(torch.tensor([[-1.0, -1.0]], device=device))
                    continue

                ct = torch.tensor([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], device=device)
                ct_trans = transform_preds_with_trans_torch(ct, trans_original_b.clone())
                location = unproject_2d_to_3d_torch(ct_trans, depth, calib_b)
                dist_thresh = get_dist_thresh_torch(calib_b, ct_trans, dim, alpha, opt, phase=phase, location=location, device=device)

                w = bbox[2] - bbox[0]
                expand_pixels = w * (opt.frustum_expand_x + opt.dynamicFrustumExpansionRatio * depth.item()**2)
                bbox = bbox.clone()  # Make a new tensor to avoid in-place ops on a view
                bbox[0] = bbox[0] - expand_pixels / 2
                bbox[2] = bbox[2] + expand_pixels / 2

                bbox_int = torch.tensor([torch.floor(bbox[0]), torch.floor(bbox[1]),
                                        torch.ceil(bbox[2]), torch.ceil(bbox[3])], dtype=torch.int32)

                roi = pc_hm_b[:, bbox_int[1]:bbox_int[3]+1, bbox_int[0]:bbox_int[2]+1]
                pc_dep = roi[opt.pc_feat_channels['pc_dep']]
                nonzero_inds = torch.nonzero(pc_dep > 0, as_tuple=True)

                if len(nonzero_inds[0]) == 0:
                    all_means.append(torch.zeros(1, 2, device=device))
                    all_covs.append(torch.zeros(1, 2, 2, device=device))
                    all_valids.append(False)
                    matched_gt_coords.append(torch.tensor([[-1.0, -1.0]], device=device))
                    continue

                xs_roi = nonzero_inds[1].float() + bbox_int[0].float()
                depths_roi = pc_dep[nonzero_inds[0], nonzero_inds[1]]
                coords = torch.stack([xs_roi, depths_roi], dim=1)

                if opt.use_dist_for_frustum:
                    pc_pos_x = roi[-2]
                    pc_pos_z = roi[-1]
                    pc_dist = torch.sqrt(torch.square(pc_pos_x) + torch.square(pc_pos_z))
                    pc_dist = pc_dist[nonzero_inds[0], nonzero_inds[1]]
                    within_thresh = (pc_dist < dist_thresh[1]) & (pc_dist > dist_thresh[0])
                else:
                    within_thresh = (depths_roi < depth + dist_thresh) & (depths_roi > max(0, depth - dist_thresh))

                coords = coords[within_thresh].unsqueeze(0)
                # print(f"Coords = {coords}")

                if coords.shape[1] == 0:
                    all_means.append(torch.zeros(1, 2, device=device))
                    all_covs.append(torch.zeros(1, 2, 2, device=device))
                    all_valids.append(False)
                    matched_gt_coords.append(torch.tensor([[-1.0, -1.0]], device=device))
                    continue

                # ✅ NOW check GT center-x and depth inclusion
                # print(f"bbox pred is {bbox}")
                # print(f"gt_xs is {gt_xs}")
                # print(f"gt_zs is {gt_zs}")
                bbox_xmin = bbox[0].item()
                bbox_xmax = bbox[2].item()
                min_z = torch.clamp(depth - dist_thresh, min=0.0)
                max_z = depth + dist_thresh
                inside_x = (gt_xs >= bbox_xmin) & (gt_xs <= bbox_xmax)
                inside_z = (gt_zs >= min_z) & (gt_zs <= max_z)
                valid_frustum = torch.any(inside_x & inside_z)
                all_valids.append(valid_frustum.item())

                if valid_frustum:
                    # Find index of first GT that is inside
                    matching = (inside_x & inside_z).nonzero(as_tuple=False)
                    if len(matching) > 0:
                        matched_idx = matching[0].item()  # Take the first matching GT
                        gt_center_x = gt_xs[matched_idx].unsqueeze(0)  # (1,)
                        gt_depth = gt_zs[matched_idx].unsqueeze(0)     # (1,)
                        # print(f"Mean GT is {gt_center_x, gt_depth}")
                        matched_gt_coords.append(torch.stack([gt_center_x, gt_depth], dim=1))  # (1, 2)
                    else:
                        matched_gt_coords.append(torch.tensor([[-1.0, -1.0]], device=device))  # Or something invalid
                else:
                    matched_gt_coords.append(torch.tensor([[-1.0, -1.0]], device=device))

                mean, cov = self.forward(coords)
                # print(f"Mean pred is {mean}")
                all_means.append(mean)
                all_covs.append(cov)
                diff = coords[0] - mean[0]
                inv_cov = torch.inverse(cov[0])
                weights = torch.exp(-0.5 * (diff @ inv_cov * diff).sum(dim=1))
                weights_sum = weights.sum()
                if weights_sum < 1e-8:
                    weights_sum = torch.tensor(1e-8, device=weights.device)
                weights = weights / weights_sum
                weights = weights / (weights.sum() + 1e-6)

                weighted_depth = (weights * coords[0][:, 1]).sum()
                if opt.normalize_depth:
                    weighted_depth /= opt.max_pc_depth

                w_interval = opt.hm_to_box_ratio * (bbox[2] - bbox[0])
                h_interval = opt.hm_to_box_ratio * (bbox[3] - bbox[1])
                w_min = int(ct[0] - w_interval / 2.)
                w_max = int(ct[0] + w_interval / 2.)
                h_min = int(ct[1] - h_interval / 2.)
                h_max = int(ct[1] + h_interval / 2.)

                H, W = pc_box_hm.shape[2], pc_box_hm.shape[3]
                h_min = max(0, h_min)
                h_max = min(H - 1, h_max)
                w_min = max(0, w_min)
                w_max = min(W - 1, w_max)

                pc_box_hm[i, opt.pc_feat_channels['pc_dep'], h_min:h_max+1, w_min:w_max+1] = weighted_depth

        final_mean = torch.cat(all_means, dim=0).unsqueeze(0)
        final_cov = torch.cat(all_covs, dim=0).unsqueeze(0)
        final_valid_mask = torch.tensor(all_valids, dtype=torch.bool, device=device).unsqueeze(0)  # [1, K]
        final_gt_coords = torch.cat(matched_gt_coords, dim=0).unsqueeze(0)  # (1, K, 2)

        return pc_box_hm, final_mean, final_cov, final_valid_mask, final_gt_coords
