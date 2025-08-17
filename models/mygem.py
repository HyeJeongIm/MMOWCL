import logging
import copy
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from models.mmeabase import MMEABaseLearner
from utils.toolkit import tensor2numpy
from models.baseline_tbn import TBNBaseline
from models.baseline_tsn import TSNBaseline

try:
    from quadprog import solve_qp
except:
    def solve_qp(*args, **kwargs):
        # Fallback implementation or simple approximation
        logging.warning("quadprog not available, using approximation")
        return [np.zeros(args[1].shape[0]), None]

EPSILON = 1e-8


class MyGEM(MMEABaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._num_segments = args["num_segments"]
        self.previous_data = None
        self.previous_label = None

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self.total_classnum = data_manager.get_total_classnum()
        
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._classes_seen_so_far = self._total_classes
        self.class_increments.append([self._known_classes, self._total_classes - 1])
        
        self._network.update_fc(self._total_classes)
        logging.info(f"Learning on {self._known_classes}-{self._total_classes}")

        # Setup data loaders using overridden method
        self._setup_data_loaders_with_ood(data_manager)

        # Prepare previous task data for gradient constraint
        if self._cur_task > 0:
            previous_dataset = data_manager.get_dataset(
                [], source="train", mode="train", appendent=self._get_memory()
            )

            self.previous_data = []
            self.previous_label = []
            for item in previous_dataset:
                _, data_, label_ = item
                # Handle multimodal data
                if isinstance(data_, dict):
                    # Convert multimodal data to appropriate format
                    processed_data = {}
                    for modality in self._modality:
                        if modality in data_:
                            processed_data[modality] = data_[modality]
                    self.previous_data.append(processed_data)
                else:
                    self.previous_data.append(data_)
                self.previous_label.append(label_)
            
            # Convert to tensor format for multimodal data
            if len(self.previous_data) > 0 and isinstance(self.previous_data[0], dict):
                # Stack multimodal data separately
                stacked_data = {}
                for modality in self._modality:
                    modality_data = [item[modality] for item in self.previous_data if modality in item]
                    if modality_data:
                        stacked_data[modality] = torch.stack(modality_data)
                self.previous_data = stacked_data
            else:
                self.previous_data = torch.stack(self.previous_data) if self.previous_data else None
            
            self.previous_label = torch.tensor(self.previous_label) if self.previous_label else None

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
            
    def _setup_data_loaders_with_ood(self, data_manager):
        """Setup train/test/ood data loaders"""
        logging.info(f"Setting up data loaders for Task {self._cur_task}")
        
        # Training data: current task classes only
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train"
            # NOTE: No appendent=self._get_memory() for GEM!
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers
        )
        
        # Test data: all seen classes so far  
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), 
            source="test", 
            mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers
        )
        
        # OOD Test data: unseen classes
        self.ood_test_loader = None
        if getattr(self, "enable_ood", True):
            if self._total_classes < self.total_classnum:
                ood_test_dataset = data_manager.get_dataset(
                    np.arange(self._total_classes, self.total_classnum),
                    source="test",
                    mode="test",
                )
                self.ood_test_loader = DataLoader(
                    ood_test_dataset,
                    batch_size=self._batch_size,
                    shuffle=False,
                    num_workers=self._num_workers,
                )
                logging.info(f"  OOD enabled. OOD classes: {self._total_classes} ~ {self.total_classnum-1}")
                logging.info(f"  OOD test samples: {len(ood_test_dataset)}")
            else:
                logging.info("  OOD enabled, but no unseen classes remain (final task).")
        else:
            logging.info("  OOD disabled by parser (enable_ood=False). Skipping OOD loader creation.")

        logging.info(f"  Train samples: {len(train_dataset)}")
        logging.info(f"  ID test samples: {len(test_dataset)}")

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        """GEM's gradient episodic memory algorithm"""
        optimizers = optimizer if isinstance(optimizer, (list, tuple)) else [optimizer]
        schedulers = scheduler if isinstance(scheduler, (list, tuple)) else [scheduler]
        
        # Calculate gradient dimensions
        grad_numels = []
        for params in self._network.parameters():
            if params.requires_grad:
                grad_numels.append(params.data.numel())
        
        # Gradient matrix: [total_params, num_tasks]
        G = torch.zeros((sum(grad_numels), self._cur_task + 1)).to(self._device)
        
        prog_bar = tqdm(range(self._epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            
            if self._partialbn:
                self._network.backbone.freeze_fn('partialbn_statistics')
            if self._freeze:
                self._network.backbone.freeze_fn('bn_statistics')
            
            losses = 0.0
            correct, total = 0, 0
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                if self.args["debug_mode"] and i >= 5:
                    break
                
                # Step 1: Calculate gradients for previous tasks (constraints)
                incremental_step = self._total_classes - self._known_classes
                
                if self._cur_task > 0 and self.previous_data is not None:
                    for k in range(0, self._cur_task):
                        for opt in optimizers:
                            opt.zero_grad()
                        
                        # Get data for task k
                        mask = torch.where(
                            (self.previous_label >= k * incremental_step) &
                            (self.previous_label < (k + 1) * incremental_step)
                        )[0]
                        
                        if len(mask) > 0:
                            if isinstance(self.previous_data, dict):
                                # Multimodal data
                                data_k = {}
                                for modality in self._modality:
                                    if modality in self.previous_data:
                                        data_k[modality] = self.previous_data[modality][mask].to(self._device)
                            else:
                                data_k = self.previous_data[mask].to(self._device)
                            
                            label_k = self.previous_label[mask].to(self._device)
                            
                            # Forward pass for previous task
                            pred_k = self._network(data_k)["logits"]
                            
                            # Mask out other tasks' outputs
                            pred_k[:, :k * incremental_step].data.fill_(-10e10)
                            pred_k[:, (k + 1) * incremental_step:].data.fill_(-10e10)
                            
                            loss_k = F.cross_entropy(pred_k, label_k)
                            loss_k.backward()
                            
                            # Store gradients in G matrix
                            j = 0
                            for params in self._network.parameters():
                                if params.requires_grad and params.grad is not None:
                                    if j == 0:
                                        stpt = 0
                                    else:
                                        stpt = sum(grad_numels[:j])
                                    
                                    endpt = sum(grad_numels[:j + 1])
                                    G[stpt:endpt, k].data.copy_(params.grad.data.view(-1))
                                    j += 1
                        
                        for opt in optimizers:
                            opt.zero_grad()
                
                # Step 2: Calculate gradient for current task
                for m in self._modality:
                    inputs[m] = inputs[m].to(self._device)
                targets = targets.to(self._device)
                
                logits = self._network(inputs)["logits"]
                
                # Mask out previous tasks' outputs for current task learning
                if self._cur_task > 0:
                    logits[:, :self._known_classes].data.fill_(-10e10)
                
                loss_clf = F.cross_entropy(logits, targets)
                loss = loss_clf
                
                for opt in optimizers:
                    opt.zero_grad()
                loss.backward()
                
                # Store current task gradient
                j = 0
                for params in self._network.parameters():
                    if params.requires_grad and params.grad is not None:
                        if j == 0:
                            stpt = 0
                        else:
                            stpt = sum(grad_numels[:j])
                        
                        endpt = sum(grad_numels[:j + 1])
                        G[stpt:endpt, self._cur_task].data.copy_(params.grad.data.view(-1))
                        j += 1
                
                # Step 3: Check for gradient conflicts and solve QP if needed
                if self._cur_task > 0:
                    dotprod = torch.mm(
                        G[:, self._cur_task].unsqueeze(0), G[:, :self._cur_task]
                    )
                    
                    if (dotprod < 0).sum() > 0:
                        # Solve quadratic programming problem
                        old_grad = G[:, :self._cur_task].cpu().t().double().numpy()
                        cur_grad = G[:, self._cur_task].cpu().contiguous().double().numpy()
                        
                        try:
                            C = old_grad @ old_grad.T
                            p = old_grad @ cur_grad
                            A = np.eye(old_grad.shape[0])
                            b = np.zeros(old_grad.shape[0])
                            
                            v = solve_qp(C, -p, A, b)[0]
                            
                            new_grad = old_grad.T @ v + cur_grad
                            new_grad = torch.tensor(new_grad).float().to(self._device)
                            
                            # Verify constraints are satisfied
                            new_dotprod = torch.mm(
                                new_grad.unsqueeze(0), G[:, :self._cur_task]
                            )
                            if (new_dotprod < -0.01).sum() > 0:
                                logging.warning("GEM constraints not satisfied, using original gradient")
                            else:
                                # Apply corrected gradient
                                j = 0
                                for params in self._network.parameters():
                                    if params.requires_grad and params.grad is not None:
                                        if j == 0:
                                            stpt = 0
                                        else:
                                            stpt = sum(grad_numels[:j])
                                        
                                        endpt = sum(grad_numels[:j + 1])
                                        params.grad.data.copy_(
                                            new_grad[stpt:endpt]
                                            .contiguous()
                                            .view(params.grad.data.size())
                                        )
                                        j += 1
                        except Exception as e:
                            logging.warning(f"QP solver failed: {e}, using original gradient")
                
                # Gradient clipping
                if self._clip_gradient is not None:
                    nn.utils.clip_grad_norm_(self._network.parameters(), self._clip_gradient)
                
                # Update parameters
                for opt in optimizers:
                    opt.step()
                
                losses += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += preds.eq(targets).sum().item()
                total += targets.numel()
            
            # Scheduler step
            for sch in schedulers:
                sch.step()
            
            train_acc = round((correct * 100.0) / max(1, total), 2)
            
            # Wandb logging
            if self.args["use_wandb"]:
                wandb.log({
                    "Train/train_loss": losses / len(train_loader),
                    "Train/train_accuracy": train_acc
                })
            
            info = f"Task {self._cur_task}, Epoch {epoch+1}/{self._epochs} => Loss {losses/len(train_loader):.3f}, Train_accy {train_acc:.2f}"
            if self.args.get("log_test_acc", False) and epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info += f", Test_accy {test_acc:.2f}"
                if self.args["use_wandb"]:
                    wandb.log({"Train/test_accuracy": test_acc})
            
            prog_bar.set_description(info)
        
        logging.info(info)

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            for m in self._modality:
                _inputs[m] = _inputs[m].to(self._device)
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._consensus(self._network.module.extract_vector(_inputs))
                )
            else:
                _vectors = tensor2numpy(
                    self._consensus(self._network.extract_vector(_inputs))
                )

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _consensus(self, x):
        output = x.view((-1, self._num_segments) + x.size()[1:])
        output = output.mean(dim=1, keepdim=True)
        output = output.squeeze(1)
        return output

    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        )
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt)
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean
            
    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                # print(mu_p)
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    data[i]
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    vectors[i]
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)
            
            self._class_means[class_idx, :] = mean


class TBN_GEM(MyGEM):
    """MyGEM model with TBN backbone"""
    
    def __init__(self, args):
        super().__init__(args)
        self._network = TBNBaseline(args)


class TSN_GEM(MyGEM):
    """MyGEM model with TSN backbone"""
    
    def __init__(self, args):
        super().__init__(args)
        self._network = TSNBaseline(args)
