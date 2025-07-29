import logging
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseLearner
from models.baseline import Baseline
from utils.toolkit import target2onehot, tensor2numpy
from ood import MSPDetector, EnergyDetector, ODINDetector
from ood.metrics import compute_ood_metrics, compute_threshold_accuracy

# EWC hyperparameters
EPSILON = 1e-8
T = 2
lamda = 1000
fishermax = 0.0001


class MyEWC(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        
        self.args = args
        self._batch_size = args["batch_size"]
        self._num_workers = args["workers"]
        self._lr = args["lr"]
        self._epochs = args["epochs"]
        self._momentum = args["momentum"]
        self._weight_decay = args["weight_decay"]
        self._lr_steps = args["lr_steps"]
        self._modality = args["modality"]

        self._partialbn = args["partialbn"]
        self._freeze = args["freeze"]
        self._clip_gradient = args["clip_gradient"]

        self.fisher = None
        self._network = Baseline(args)
        self.class_increments = []
        
    def after_task(self):
        """Update known classes after task completion"""
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        """Main incremental training function"""
        self.total_classnum = data_manager.get_total_classnum()
        
        # Update task state
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        
        self._classes_seen_so_far = self._total_classes
        self.class_increments.append([self._known_classes, self._total_classes-1])
        
        # Update classifier for new classes
        self._network.update_fc(self._total_classes)
        logging.info("Learning on classes {}-{}".format(self._known_classes, self._total_classes-1))

        # Setup data loaders
        self._setup_data_loaders_with_ood(data_manager)

        # Multi-GPU setup
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # Update Fisher information matrix
        if self.fisher is None:
            self.fisher = self.getFisherDiagonal(self.train_loader)
        else:
            alpha = self._known_classes / self._total_classes
            new_finsher = self.getFisherDiagonal(self.train_loader)
            for n, p in new_finsher.items():
                new_finsher[n][: len(self.fisher[n])] = (
                        alpha * self.fisher[n]
                        + (1 - alpha) * new_finsher[n][: len(self.fisher[n])]
                )
            self.fisher = new_finsher
        self.mean = {
            n: p.clone().detach()
            for n, p in self._network.named_parameters()
            if p.requires_grad
        }

    def _setup_data_loaders_with_ood(self, data_manager):
        """Setup train/test/ood data loaders"""
        logging.info(f"Setting up data loaders for Task {self._cur_task}")
        
        # Training data: current task classes only
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
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
        
        # OOD test data: unseen classes
        if self._total_classes < self.total_classnum:
            ood_test_dataset = data_manager.get_dataset(
                np.arange(self._total_classes, self.total_classnum),
                source="test",
                mode="test"
            )
            self.ood_test_loader = DataLoader(
                ood_test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers
            )
            logging.info(f"  OOD classes: {self._total_classes} ~ {self.total_classnum-1}")
        else:
            self.ood_test_loader = None
            logging.info("  No OOD classes available (final task)")
        
        logging.info(f"  Train samples: {len(train_dataset)}")
        logging.info(f"  ID test samples: {len(test_dataset)}")
        if self.ood_test_loader:
            logging.info(f"  OOD test samples: {len(ood_test_dataset)}")

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        optimizer = self._choose_optimizer()

        # Setup scheduler
        if type(optimizer) == list:
            scheduler_adam = optim.lr_scheduler.MultiStepLR(optimizer[0], self._lr_steps, gamma=0.1)
            scheduler_sgd = optim.lr_scheduler.MultiStepLR(optimizer[1], self._lr_steps, gamma=0.1)
            scheduler = [scheduler_adam, scheduler_sgd]
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, self._lr_steps, gamma=0.1)

        if self._cur_task == 0:
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
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
                for m in self._modality:
                    inputs[m] = inputs[m].to(self._device)
                targets = targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)

                if type(optimizer) == list:
                    optimizer[0].zero_grad()
                    optimizer[1].zero_grad()
                else:
                    optimizer.zero_grad()

                loss.backward()

                if self._clip_gradient is not None:
                    total_norm = nn.utils.clip_grad_norm_(self._network.parameters(), self._clip_gradient)

                if type(optimizer) == list:
                    optimizer[0].step()
                    optimizer[1].step()
                else:
                    optimizer.step()

                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if type(scheduler) == list:
                scheduler[0].step()
                scheduler[1].step()
            else:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self._epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self._epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        """Incremental training with EWC regularization"""
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
                for m in self._modality:
                    inputs[m] = inputs[m].to(self._device)
                targets = targets.to(self._device)
                logits = self._network(inputs)["logits"]

                # Classification loss + EWC regularization
                loss_clf = F.cross_entropy(
                    logits[:, self._known_classes:], targets - self._known_classes
                )
                loss_ewc = self.compute_ewc()
                loss = loss_clf + lamda * loss_ewc

                if type(optimizer) == list:
                    optimizer[0].zero_grad()
                    optimizer[1].zero_grad()
                else:
                    optimizer.zero_grad()

                loss.backward()

                if self._clip_gradient is not None:
                    total_norm = nn.utils.clip_grad_norm_(self._network.parameters(), self._clip_gradient)

                if type(optimizer) == list:
                    optimizer[0].step()
                    optimizer[1].step()
                else:
                    optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if type(scheduler) == list:
                scheduler[0].step()
                scheduler[1].step()
            else:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self._epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
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

    def compute_ewc(self):
        """Compute EWC regularization loss"""
        loss = 0
        if len(self._multiple_gpus) > 1:
            for n, p in self._network.module.named_parameters():
                if n in self.fisher.keys():
                    loss += (
                            torch.sum(
                                (self.fisher[n])
                                * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                            )
                            / 2
                    )
        else:
            for n, p in self._network.named_parameters():
                if n in self.fisher.keys():
                    loss += (
                            torch.sum(
                                (self.fisher[n])
                                * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                            )
                            / 2
                    )
        return loss

    def getFisherDiagonal(self, train_loader):
        fisher = {
            n: torch.zeros(p.shape).to(self._device)
            for n, p in self._network.named_parameters()
            if p.requires_grad
        }
        self._network.train()
        optimizer = optim.SGD(self._network.parameters(), lr=self._lr)
        for i, (_, inputs, targets) in enumerate(train_loader):
            for m in self._modality:
                inputs[m] = inputs[m].to(self._device)
            targets = targets.to(self._device)
            logits = self._network(inputs)["logits"]
            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            for n, p in self._network.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2).clone()
        for n, p in fisher.items():
            fisher[n] = p / len(train_loader)
            fisher[n] = torch.min(fisher[n], torch.tensor(fishermax))
        return fisher

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            for m in self._modality:
                inputs[m] = inputs[m].to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            for m in self._modality:
                inputs[m] = inputs[m].to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
    
    def evaluate_cl_ood(self):
        """Evaluate both CL accuracy and OOD detection performance"""
        logging.info(f"=== Task {self._cur_task} Evaluation ===")
        logging.info(f"Known classes: 0-{self._classes_seen_so_far-1}")
        logging.info(f"Unknown classes: {self._classes_seen_so_far}-{self.total_classnum-1}")
        
        # Step 1: Standard CL accuracy evaluation
        cnn_accy, nme_accy = self.eval_task()
        '''
            cnn_accy: {'grouped': {'00-09': 81.27}, 'top1': 81.27}
            nme_accy: None
        '''
        
        if nme_accy is not None:
            logging.info(f"CL Accuracy - CNN: {cnn_accy['top1']:.2f}%, NME: {nme_accy['top1']:.2f}%")
        else:
            logging.info(f"CL Accuracy - CNN: {cnn_accy['top1']:.2f}%, NME: Not Available")
        
        # Step 2: Multiple OOD method evaluation
        if "ood_methods" not in self.args:
            logging.error("ood_methods not found in configuration file!")
            return {}, {'cnn': cnn_accy, 'nme': nme_accy if nme_accy else {'top1': 0.0, 'grouped': {}}}
        
        ood_methods = self.args["ood_methods"]
        
        logging.info(f"OOD Methods from JSON: {ood_methods}")
        if self.ood_test_loader is None:
            logging.warning("No OOD test data available. Skipping OOD evaluation.")
            return {}, {'cnn': cnn_accy, 'nme': nme_accy if nme_accy else {'top1': 0.0, 'grouped': {}}}
        
        ood_results = {}
        
        logging.info("=== OOD Detection Results ===")
        
        for method_name in ood_methods:
            try:
                # Initialize OOD detector
                if method_name == "MSP":
                    detector = MSPDetector(self._network, self._device)
                elif method_name == "Energy":
                    detector = EnergyDetector(self._network, self._device)
                elif method_name == "ODIN":
                    detector = ODINDetector(self._network, self._device)
                else:
                    logging.warning(f"Unknown OOD method: {method_name}")
                    continue
                
                logging.info(f"Computing {method_name} scores...")
                
                # Compute OOD scores
                id_scores = detector.compute_scores(self.test_loader)      
                ood_scores = detector.compute_scores(self.ood_test_loader) 
                
                # Compute OOD metrics
                metrics = compute_ood_metrics(id_scores, ood_scores, method_name)
                ood_results[method_name] = metrics
                
                # Log results
                if 'error' not in metrics:
                    logging.info(f"{method_name}: AUROC={metrics['auroc']:.2f}%, FPR95={metrics['fpr95']:.2f}%")
                    logging.info(f"  Samples - ID: {metrics['id_samples']}, OOD: {metrics['ood_samples']}")
                else:
                    logging.error(f"{method_name}: Error - {metrics['error']}")
                    
            except Exception as e:
                logging.error(f"{method_name} evaluation failed: {e}")
                ood_results[method_name] = {'error': str(e), 'method': method_name}
        
        # Store results for trainer access
        self.latest_ood_results = ood_results
        self.latest_cl_results = {'cnn': cnn_accy, 'nme': nme_accy}
        
        return ood_results, {'cnn': cnn_accy, 'nme': nme_accy}