import copy
import logging
import numpy as np
import random
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from models.baseline_tsn import TSNBaseline
from utils.toolkit import count_parameters, tensor2numpy
from ood import MSPDetector, EnergyDetector, ODINDetector
from ood.metrics import compute_ood_metrics, compute_threshold_accuracy
from models.mmeabase import MMEABaseLearner


EPSILON = 1e-8
T = 2


class CMR_MFN(MMEABaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._batch_size = args["batch_size"]
        self._num_workers = args["workers"]
        self._lr = args["lr"]
        self._epochs = args["epochs"]
        self._momentum = args["momentum"]
        self._weight_decay = args["weight_decay"]
        self._lr_steps = args["lr_steps"]
        self._modality = args["modality"]

        self._freeze = args["freeze"]
        self._clip_gradient = args["clip_gradient"]

        self._network = TSNBaseline(args)
        
        # OOD related attributes
        self.args = args
        self.total_classnum = None
        self.ood_test_loader = None
        self.test_loader = None
        self.train_loader = None
        self._classes_seen_so_far = 0
        self.class_increments = []
        
    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._cur_task_size = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._cur_task_size
        self.data_manager = data_manager
        
        # Set total class number for OOD evaluation
        if self.total_classnum is None:
            self.total_classnum = data_manager.get_total_classnum()
        self._classes_seen_so_far = self._total_classes
        
        # Update class increments for accuracy calculation
        self.class_increments.append([self._known_classes, self._total_classes-1])

        # Update classifier for current task using gen_train_fc method
        self._network.gen_train_fc(self._cur_task_size * 2)

        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        # B버전: 이전 태스크의 파라미터 freeze
        if self._cur_task > 0:
            for i in range(self._cur_task):
                for p in self._network.fusion_networks[i].parameters():
                    p.requires_grad = False
                for p in self._network.fc_list[i].parameters():
                    p.requires_grad = False

        # Setup data loaders with OOD support
        self._setup_data_loaders_with_ood(data_manager)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)


    def train(self):
        self._network.train()
        # B버전: 이전 태스크들은 eval 모드로 설정
        if self._cur_task > 0:
            for i in range(self._cur_task):
                self._network.fusion_networks[i].eval()
                self._network.fc_list[i].eval()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self._network.parameters()),
                                        self._lr,
                                        weight_decay=self._weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, self._lr_steps, gamma=0.1)
        if self._cur_task == 0:
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
        # B버전: 현재 태스크 파라미터 저장
        self._network.save_parameter()

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self._epochs))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                for m in self._modality:
                    inputs[m] = inputs[m].to(self._device)
                targets = targets.to(self._device)

                # Extract features using Baseline structure
                features = self._network.backbone(inputs)
                fake_inputs, fake_targets = self._confusion_mixup(features, targets)
                fusion_output = self._network.fusion_network(fake_inputs)
                fake_logits = self._network.fc(fusion_output["features"])['logits']
                
                loss_clf = F.cross_entropy(fake_logits, fake_targets)
                loss = loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(fake_logits, dim=1)
                correct += preds.eq(fake_targets.expand_as(preds)).cpu().sum()
                total += len(fake_targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            self.training_iterations += 1

            if epoch % 5 == 0:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self._epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self._epochs,
                    losses / len(train_loader),
                    train_acc,

                )
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self._epochs))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                for m in self._modality:
                    inputs[m] = inputs[m].to(self._device)
                targets = targets.to(self._device)
                targets = targets - self._known_classes

                # Extract features using Baseline structure
                features = self._network.backbone(inputs)
                fake_inputs, fake_targets = self._confusion_mixup(features, targets)
                fusion_output = self._network.fusion_network(fake_inputs)
                fake_logits = self._network.fc(fusion_output["features"])['logits']
                
                loss_clf = F.cross_entropy(fake_logits, fake_targets)
                loss = loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(fake_logits, dim=1)
                correct += preds.eq(fake_targets.expand_as(preds)).cpu().sum()
                total += len(fake_targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            self.training_iterations += 1
            if epoch % 5 == 0:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self._epochs,
                    losses / len(train_loader),
                    train_acc,

                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self._epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def _eval_cnn(self, loader):
        self._network.fusion_networks.to(self._device)
        self._network.fc_list.to(self._device)
        self._network.eval()
        y_pred, y_true = [], []
        results = []
        for _, (_, inputs, targets) in enumerate(loader):
            for m in self._modality:
                inputs[m] = inputs[m].to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs, self._cur_task_size, mode='test')
                logits = outputs["logits"]
            predicts = torch.topk(
                logits, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

            results.append({'features': {m: outputs['features'][m].cpu().numpy() for m in self._modality},
                            'fusion_features': outputs['fusion_features'].cpu().numpy(),
                            'logits': logits.cpu().numpy()})

        return np.concatenate(y_pred), np.concatenate(y_true), results  # [N, topk]

    def eval_task(self, scores_dir):
        y_pred, y_true, results = self._eval_cnn(self.test_loader)
        self.save_scores(results, y_true, y_pred, '{}/{}.pkl'.format(scores_dir, self._cur_task))
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        return cnn_accy, nme_accy
    
    def save_scores(self, results, y_true, y_pred, filename):
        """Save evaluation results to file"""
        import pickle
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        save_data = {
            'results': results,
            'y_true': y_true,
            'y_pred': y_pred,
            'task': self._cur_task
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)

    def _map_targets(self, select_targets):
        mixup_targets = select_targets + self._cur_task_size
        return mixup_targets

    def _confusion_mixup(self, inputs, targets, alpha=0.2, mix_time=2):
        mixup_inputs = {}
        for m in self._modality:
            mixup_inputs[m] = []
        mixup_targets = []

        for _ in range(mix_time):
            index = torch.randperm(inputs[self._modality[0]].shape[0])
            perm_targets = targets[index]

            mask = perm_targets != targets
            
            for m in self._modality:
                select_inputs = inputs[m][mask]
                perm_inputs = inputs[m][index][mask]
                
                lams = np.random.beta(alpha, alpha, size=sum(mask))
                lams = np.where(lams < 0.5, 0.75, lams)
                # lams = torch.from_numpy(lams).cuda(4)[:, None].float()
                lams = torch.from_numpy(lams).to(inputs[m].device)[:, None].float()        
                        
                if len(lams) != 0:
                    mixup_input = torch.cat(
                        [torch.unsqueeze(lams[i] * select_inputs[i] + (1 - lams[i]) * perm_inputs[i], 0) for i in
                         range(len(lams))], 0)
                    
                    mixup_inputs[m].append(mixup_input)

            if len(lams) != 0:
                select_targets = targets[mask]
                perm_targets = perm_targets[mask]
                mixup_targets.append(self._map_targets(select_targets))        

        for m in self._modality:
            if len(mixup_inputs[m]) != 0:
                mixup_inputs[m] = torch.cat(mixup_inputs[m], dim=0)
                inputs[m] = torch.cat([inputs[m], mixup_inputs[m]], dim=0)
        
        if len(mixup_targets) != 0:
            mixup_targets = torch.cat(mixup_targets, dim=0)
            targets = torch.cat([targets, mixup_targets], dim=0)

        return inputs, targets
