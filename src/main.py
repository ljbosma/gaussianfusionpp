from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import shutil

import torch
import torch.utils.data

from opts import opts
from model.model import create_model, freeze_layers, load_model, save_model, merge_models
from logger import Logger
from dataset.dataset_factory import get_dataset
from trainer import Trainer
import json
from utils.eval_frustum import EvalFrustum


def get_optimizer(opt, model):
  if opt.optim == 'adam':
    print('Using adam.')
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  elif opt.optim == 'sgd':
    print('Using SGD')
    optimizer = torch.optim.SGD(
      model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay) 
  else:
    raise ValueError("Optimizer not implemented yet.")
  return optimizer

def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.eval
  Dataset = get_dataset(opt.dataset)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print('-'*100, '\nOPTIONS:')
  print(opt, '\n', '-'*100)
  if not opt.not_set_cuda_env:
    # Not training on GPU cluster
    print('Setting CUDA_VISIBLE_DEVICES.')
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str   # e.g. "0,1"
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  logger = Logger(opt)
  loss_keys_2d = ['hm', 'wh', 'reg']
  loss_keys_3d = ['dep', 'dep_sec', 'dim', 'rot', 'rot_sec', 'amodel_offset', 'nuscenes_att', 'velocity', 'rgpnet']

  # Get and print information about current GPU setup
  if torch.cuda.is_available():
    print('CUDA is available.') 
    num_gpus = torch.cuda.device_count()
    print(f'{num_gpus} GPU(s) were found.')
    vis_dev = os.environ.get('CUDA_VISIBLE_DEVICES')
    print(f'CUDA_VISIBLE_DEVICES is set to: {vis_dev}')
    for idx in range(num_gpus):
      print(f'Properties of GPU with index {idx}:\n{torch.cuda.get_device_properties(idx)}')
  else:
    print('CUDA is NOT available.')

  if opt.merge_models:
    template_model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
    model, epoch = merge_models(template_model, opt.load_model, opt.load_model2, opt)
    optimizer = get_optimizer(opt, model)
    print('Merged model is saved in:', os.path.join(opt.log_dir, f'model_{epoch}.pth'))
    save_model(os.path.join(opt.log_dir, f'model_{epoch}.pth'), 
                  epoch, model, optimizer)
  else:
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv, local_pretrained_path="../models/centernet_model_best_20_epochs.pth", opt=opt)
    optimizer = get_optimizer(opt, model)
    start_epoch = 0
    lr = opt.lr

  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, opt, optimizer)
  
  if opt.freeze_layers:
    model = freeze_layers(model, opt)

  if opt.set_eval_layers:
    for layer in opt.layers_to_eval:
        print('Layer set to eval()-mode in training: ', layer)

  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  # Create object of class EvalFrustum if required
  if opt.eval_frustum > 0:
    eval_frustum = EvalFrustum(opt)
  else:
    eval_frustum = None

  if opt.val_intervals <= opt.num_epochs or opt.eval:
    print('Setting up validation data...')
    # Set the batchsize to the number of GPUs used
    val_loader = torch.utils.data.DataLoader(
      Dataset(opt=opt, split=opt.val_split, eval_frustum=eval_frustum), batch_size=1, shuffle=False, 
              num_workers=opt.num_workers, pin_memory=True)

    if opt.eval:
      # If eval is active then just validate and not train
      _, preds = trainer.val(0, val_loader)
      val_loader.dataset.run_eval(preds, opt.log_dir, n_plots=opt.eval_n_plots, 
                                  render_curves=opt.eval_render_curves)
      return

  print('Setting up training data...')
  train_loader = torch.utils.data.DataLoader(
      Dataset(opt=opt, split=opt.train_split, eval_frustum=eval_frustum), batch_size=opt.batch_size, 
        shuffle=opt.shuffle_train, num_workers=opt.num_workers, 
        pin_memory=True, drop_last=True
  )

  print('Starting training...')
  history = {}
  best_metric = float('inf')
  epochs_without_improvement = 0
  if not hasattr(opt, 'early_stop_patience'):
    opt.early_stop_patience = 15

  
  # Loop over epochs
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    phase1_end = 5     # Phase 1: 2D OD only
    phase2_end = 20    # Phase 2: Unfreeze, mix
    phase3_start = phase2_end + 1

    # Update loss weights based on phase
    if epoch <= phase1_end:
      opt.freeze_layers = True
      for param in model.parameters():
        param.requires_grad = True
      opt.weights.update({
          'hm': 2.0, 'wh': 1.0, 'reg': 2.0,
          'dep': 0.1, 'dep_sec': 0.1,
          'dim': 0.1, 'rot': 0.1, 'rot_sec': 0.1,
          'amodel_offset': 0.1, 'nuscenes_att': 0.1,
          'velocity': 0.1, 'rgpnet_tot': 0.1
      })
      if epoch == 1:
        print("🧊 Phase 1: Frozen backbone, strong 2D supervision")

    elif epoch <= phase2_end:
      opt.freeze_layers = False
      for param in model.parameters():
        param.requires_grad = True
      opt.weights.update({
          'hm': 1.0, 'wh': 0.5, 'reg': 1.0,
          'dep': 0.5, 'dep_sec': 0.5,
          'dim': 0.5, 'rot': 0.5, 'rot_sec': 0.5,
          'amodel_offset': 0.5, 'nuscenes_att': 0.5,
          'velocity': 0.5, 'rgpnet_tot': 0.5
      })
      if epoch == phase1_end + 1:
        print("🪄 Phase 2: Backbone unfrozen, start co-adaptation")

    else:
      opt.freeze_layers = False
      opt.weights.update({
          'hm': 0.5, 'wh': 0.1, 'reg': 0.5,
          'dep': 2.0, 'dep_sec': 2.0,
          'dim': 2.0, 'rot': 2.0, 'rot_sec': 2.0,
          'amodel_offset': 1.0, 'nuscenes_att': 1.0,
          'velocity': 2.0, 'rgpnet_tot': 2.0
      })
      if epoch == phase3_start:
        print("🚀 Phase 3: Full 3D OD focus")

    mark = epoch if opt.save_all else 'last'

    # log learning rate
    for param_group in optimizer.param_groups:
      lr = param_group['lr']
      logger.scalar_summary('LR', lr, epoch)
      break
    
    # train one epoch
    log_dict_train, _ = trainer.train(epoch, train_loader, eval_frustum=eval_frustum)
    logger.write('epoch: {} |'.format(epoch))
    
    # log train results
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    
    # Save model as "model_last.pth or model_epoch when opt.save_all"
    save_model(os.path.join(opt.log_dir, f'model_{mark}.pth'), 
                epoch, model, optimizer)

    # evaluate
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      # Validate
      with torch.no_grad():
        # Validate current epoch, create validation summary as .json
        log_dict_val, preds = trainer.val(epoch, val_loader)
        
        # evaluate val set using dataset-specific evaluator
        if opt.run_dataset_eval:
          out_dir = val_loader.dataset.run_eval(preds, opt.log_dir, 
                                                n_plots=opt.eval_n_plots, 
                                                render_curves=opt.eval_render_curves
                                                )

          # log dataset-specific evaluation metrics
          with open(os.path.join(out_dir, 'metrics_summary.json'), 'r') as f:
            metrics = json.load(f)
          
          # Save metrics and losses for this epoch
          epoch_metrics = metrics  # Save all available metrics, including NDS and COCO scores

          epoch_losses = {k: v for k, v in log_dict_val.items()}

          history[epoch] = {
              "metrics": epoch_metrics,
              "losses": epoch_losses
          }
          
          # log eval results
          for k, v in log_dict_val.items():
            logger.scalar_summary('val_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        # # === Early stopping ===
        if epoch >= phase3_start:
          current_metric = -metrics.get('NDS', 0.0)  # Maximize NDS → minimize negative
          print(f"[EarlyStopping - Phase 3] Using NDS = {metrics.get('NDS', 0.0):.6f}")

          if current_metric < best_metric:
              best_metric = current_metric
              epochs_without_improvement = 0
              save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                          epoch, model, optimizer)

              if epoch > 20:
                  # Copy debug folder to debug_best
                  debug_dir = os.path.join(opt.save_dir, 'debug')
                  debug_best_dir = os.path.join(opt.save_dir, 'debug_best')

                  if os.path.exists(debug_dir):
                      if os.path.exists(debug_best_dir):
                          shutil.rmtree(debug_best_dir)
                      shutil.copytree(debug_dir, debug_best_dir)
                      print(f"📁 Copied debug folder to {debug_best_dir}")
          else:
              epochs_without_improvement += 1
              print(f"No improvement for {epochs_without_improvement} epoch(s).")

          if epochs_without_improvement >= opt.early_stop_patience:
              print(f"Early stopping at epoch {epoch} (best NDS: {-best_metric:.4f})")
              break

    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                  epoch, model, optimizer)
  
    logger.write('\n')
    # Save the model if wanted
    if epoch in opt.save_point:
      save_model(os.path.join(opt.log_dir, f'model_{epoch}.pth'), 
                epoch, model, optimizer)
    
    # update learning rate
    if epoch in opt.lr_step:
      # Drop lr by a factor of opt.lr_step_factor at every epoch listed in list lr_step
      lr = opt.lr * (opt.lr_step_factor ** (opt.lr_step.index(epoch) + 1)) 
      print('Drop LR to', lr)
      # save lr into optimizer
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr

    # If frustum evaluation is required, save/plt the histograms
    if opt.eval_frustum > 0:
      eval_frustum.print()
      if opt.eval_frustum < 4:
        eval_frustum.plot_histograms(save_plots=False)
      if opt.eval_frustum == 4:
        eval_frustum.plot_histograms(save_plots=True)
      if opt.eval_frustum == 5:
        # Dump snapshot evaluation using pickle
        eval_frustum.dump_snapshot_eval()

  # Save full history to JSON after training
  history_path = os.path.join(opt.save_dir, 'training_history.json')
  with open(history_path, 'w') as f:
      json.dump(history, f, indent=2)

  print(f"Saved full training history to {history_path}")
  logger.write(f"Saved full training history to {history_path}\n")

  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  torch.cuda.empty_cache()
  main(opt)
