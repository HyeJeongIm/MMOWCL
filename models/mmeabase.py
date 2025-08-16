import copy
import logging
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from ood import MSPDetector, EnergyDetector, ODINDetector
from ood.metrics import compute_ood_metrics, compute_threshold_accuracy


EPSILON = 1e-8
batch_size = 64


class MMEABaseLearner(BaseLearner):
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
        self.enable_ood = args["enable_ood"]


        self.fisher = None
        self._network = None # Placeholder for the network
        self.class_increments = []

    def _setup_data_loaders_with_ood(self, data_manager):
        """Setup train/test/ood data loaders"""
        logging.info(f"Setting up data loaders for Task {self._cur_task}")
        
        # Training data: current task classes only
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(), # return None, if memory_size is 0
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
        # 3) OOD Test (parser가 허용할 때만 생성)
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
        optimizers = optimizer if isinstance(optimizer, (list, tuple)) else [optimizer]
        schedulers = scheduler if isinstance(scheduler, (list, tuple)) else [scheduler]
        
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
                
                for m in self._modality:
                    inputs[m] = inputs[m].to(self._device)
                targets = targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)

                # zero gradients
                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)

                loss.backward()

                if self._clip_gradient is not None:
                    total_norm = nn.utils.clip_grad_norm_(self._network.parameters(), self._clip_gradient)

                # optimizer step
                for opt in optimizers:
                    opt.step()

                losses += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct += preds.eq(targets).sum().item()
                total += targets.numel()

            # epoch-level scheduler step
            for sch in schedulers:
                sch.step()

            train_acc = round((correct * 100.0) / max(1, total), 2)

            # Log training metrics to W&B
            if self.args['use_wandb']:
                wandb.log({
                    "Train/train_loss": losses / len(train_loader),
                    "Train/train_accuracy": train_acc,
                })

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self._epochs,
                losses / len(train_loader),
                train_acc,
            )
            if self.args.get("log_test_acc", False) and epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info += f", Test_accy {test_acc:.2f}"
                # Log test metrics to W&B
                if self.args['use_wandb']:
                    wandb.log({
                        "Train/test_accuracy": test_acc
                    })
            
            prog_bar.set_description(info)
        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        pass

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
        # Log task metrics to W&B
        if self.args['use_wandb']:
            wandb.log({
                "Task/avg_acc": cnn_accy['top1'],
                **{f"Task/[{k}]_acc": v for k, v in cnn_accy['grouped'].items()},
            })

        if not self.enable_ood:
            logging.info("Skipping OOD evaluation (enable_ood=False).")
            return {}, {'cnn': cnn_accy, 'nme': nme_accy if nme_accy else {'top1': 0.0, 'grouped': {}}}, {}
        
        else:
            # Step 2: Multiple OOD method evaluation
            if "ood_methods" not in self.args:
                logging.error("ood_methods not found in configuration file!")
                return  {}, {'cnn': cnn_accy, 'nme': nme_accy}, {}  
                      
            ood_methods = self.args["ood_methods"]
            
            logging.info(f"OOD Methods from JSON: {ood_methods}")
            if self.ood_test_loader is None:
                logging.warning("No OOD test data available. Skipping OOD evaluation.")
                return  {}, {'cnn': cnn_accy, 'nme': nme_accy}, {}
            
            ood_results = {}
            score_distributions = {}  # Store ID/OOD scores for visualization
            
            logging.info("=== OOD Detection Results ===")
                    
            # # Get ID logits (single forward pass)
            print("  📊 Processing ID data...")
            id_logits = self._extract_logits_batch(self.test_loader)
            
            # Get OOD logits (single forward pass)  
            print("  🎯 Processing OOD data...")
            ood_logits = self._extract_logits_batch(self.ood_test_loader)
            
            print(f"✅ Logits extracted - ID: {id_logits.shape}, OOD: {ood_logits.shape}")
            
            logging.info("Computing logits once for all OOD methods...")

            for method_name in tqdm(ood_methods, desc="OOD Methods", position=0):
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
                    # id_scores = detector.compute_scores(self.test_loader)      
                    # ood_scores = detector.compute_scores(self.ood_test_loader)
                    id_scores = detector.compute_scores_from_cached_logits(id_logits)      
                    ood_scores = detector.compute_scores_from_cached_logits(ood_logits) 
                    
                    # Store score distributions for visualization
                    score_distributions[method_name] = {
                        'id_scores': id_scores.tolist() if hasattr(id_scores, 'tolist') else list(id_scores),
                        'ood_scores': ood_scores.tolist() if hasattr(ood_scores, 'tolist') else list(ood_scores)
                    }
                    
                    # Compute OOD metrics
                    metrics = compute_ood_metrics(id_scores, ood_scores, method_name)
                    ood_results[method_name] = metrics
                    
                    # Log results
                    if 'error' not in metrics:
                        logging.info(f"{method_name}: AUROC={metrics['auroc']:.2f}%, FPR95={metrics['fpr95']:.2f}%")
                        logging.info(f"  Samples - ID: {metrics['id_samples']}, OOD: {metrics['ood_samples']}")
                        logging.info(f"  ID Score Range: [{id_scores.min():.3f}, {id_scores.max():.3f}]")
                        logging.info(f"  OOD Score Range: [{ood_scores.min():.3f}, {ood_scores.max():.3f}]")
                        # Log OOD metrics to W&B
                        if self.args['use_wandb']:
                            wandb.log({
                                f"Task/{method_name}_auroc": metrics['auroc'],
                                f"Task/{method_name}_fpr95": metrics['fpr95']
                            })
                    else:
                        logging.error(f"{method_name}: Error - {metrics['error']}")
                        
                except Exception as e:
                    logging.error(f"{method_name} evaluation failed: {e}")
                    ood_results[method_name] = {'error': str(e), 'method': method_name}
            
        # Store results for trainer access
        self.latest_ood_results = ood_results
        self.latest_cl_results = {'cnn': cnn_accy, 'nme': nme_accy}
        
        return ood_results, {'cnn': cnn_accy, 'nme': nme_accy}, score_distributions
    
    def _extract_logits_batch(self, loader):
        """Extract logits from data loader in a single pass"""
        self._network.eval()
        all_logits = []
        
        with torch.no_grad():
            for _, inputs, targets in tqdm(loader, desc="Extracting logits", leave=False):
                # Handle multimodal inputs
                if isinstance(inputs, dict):
                    for m in inputs:
                        inputs[m] = inputs[m].to(self._device)
                else:
                    inputs = inputs.to(self._device)
                
                outputs = self._network(inputs)
                all_logits.append(outputs["logits"].cpu())
        
        return torch.cat(all_logits, dim=0)