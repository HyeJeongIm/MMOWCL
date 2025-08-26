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
        # import ipdb; ipdb.set_trace()
        if nme_accy is not None:
            logging.info(f"CL Accuracy - CNN: {cnn_accy['top1']:.2f}%, NME: {nme_accy['top1']:.2f}%")
        else:
            logging.info(f"CL Accuracy - CNN: {cnn_accy['top1']:.2f}%, NME: Not Available")
            
        # Log task metrics to W&B (FC 분류기, NME 분류기)
        if self.args['use_wandb']:
            wandb.log({"Task/avg_acc": cnn_accy['top1']})
            for k, v in cnn_accy['grouped'].items():
                wandb.log({f"Task/[{k}]_acc": v})

            # ── W&B 로깅 (NME, 있으면만)
            if nme_accy is not None:
                wandb.log({"Task/nme_avg_acc": nme_accy['top1']})
                for k, v in nme_accy.get('grouped', {}).items():
                    wandb.log({f"Task/NME_[{k}]_acc": v})
                    
        # if self.args['use_wandb']:
        #     wandb.log({
        #         "Task/avg_acc": cnn_accy['top1'],
        #         **{f"Task/[{k}]_acc": v for k, v in cnn_accy['grouped'].items()},
        #     })

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
                    
            # 한 번의 forward pass로 모든 데이터 추출 (logits + features)
            print("  📊 Processing ID data (logits + features)...")
            # id_data keys: dict_keys(['logits', 'features', 'labels'])
            id_data = self._extract_data_batch(self.test_loader, extract_features=True, extract_logits=True)
            id_logits = id_data['logits']
            id_features = id_data['features'] 
            id_labels = id_data['labels']
            
            print("  🎯 Processing OOD data (logits + features)...")
            ood_data = self._extract_data_batch(self.ood_test_loader, extract_features=True, extract_logits=True)
            ood_logits = ood_data['logits']
            ood_features = ood_data['features']
            ood_labels = ood_data['labels']
            
            print(f"✅ Data extracted - ID: logits{id_logits.shape}, features{id_features.shape}")
            print(f"                   OOD: logits{ood_logits.shape}, features{ood_features.shape}")
            
            logging.info("Single forward pass completed for all methods...")
            
                    # Store extracted data for T-SNE visualization (avoid re-extraction)
            self._cached_id_data = {'features': id_features, 'labels': id_labels}
            self._cached_ood_data = {'features': ood_features, 'labels': ood_labels}

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
                    
                    # Extract First Class (originally class 26) scores separately (for tracking specific class evolution)
                    class_26_indices = np.where(id_labels == 0)[0]  # First class is always 0 in remapped labels
                    class_26_scores = id_scores[class_26_indices] if len(class_26_indices) > 0 else np.array([])
                    
                    # Store score distributions for visualization
                    score_distributions[method_name] = {
                        'id_scores': id_scores.tolist() if hasattr(id_scores, 'tolist') else list(id_scores),
                        'ood_scores': ood_scores.tolist() if hasattr(ood_scores, 'tolist') else list(ood_scores),
                        'class_26_scores': class_26_scores.tolist() if hasattr(class_26_scores, 'tolist') else list(class_26_scores),
                        'class_26_count': len(class_26_indices)
                    }
                    
                    # Compute OOD metrics
                    metrics = compute_ood_metrics(id_scores, ood_scores, method_name)
                    ood_results[method_name] = metrics
                    
                    # Log results
                    if 'error' not in metrics:
                        logging.info(f"{method_name}: AUROC={metrics['auroc']:.2f}%, FPR95={metrics['fpr95']:.2f}%, AUPR={metrics['aupr_id']:.2f}%, YoudenJ={metrics['youdenJ']:.3f}")
                        logging.info(f"  AUPR Debug - Raw value: {metrics['aupr_id']}, Type: {type(metrics['aupr_id'])}")
                        logging.info(f"  Samples - ID: {metrics['id_samples']}, OOD: {metrics['ood_samples']}")
                        logging.info(f"  ID Score Range: [{id_scores.min():.3f}, {id_scores.max():.3f}]")
                        logging.info(f"  OOD Score Range: [{ood_scores.min():.3f}, {ood_scores.max():.3f}]")
                        
                        # ── Confusion@FPR95 요약 로그
                        cf = metrics['confusion_fpr95']
                        logging.info(
                            f"  Conf@FPR95 thr={cf['threshold']:.4f} | "
                            f"TP={cf['tp']} FP={cf['fp']} TN={cf['tn']} FN={cf['fn']} | "
                            f"TPR={cf['tpr']:.3f} FPR={cf['fpr']:.3f} | "
                            f"Prec={cf['precision']:.3f} Rec={cf['recall']:.3f} F1={cf['f1']:.3f}"
                        )
                        
                        # ── Confusion@YoudenJ 요약 로그
                        cf_youden = metrics['confusion_youdenJ']
                        logging.info(
                            f"  Conf@YoudenJ thr={cf_youden['threshold']:.4f} | "
                            f"TP={cf_youden['tp']} FP={cf_youden['fp']} TN={cf_youden['tn']} FN={cf_youden['fn']} | "
                            f"TPR={cf_youden['tpr']:.3f} FPR={cf_youden['fpr']:.3f} | "
                            f"Prec={cf_youden['precision']:.3f} Rec={cf_youden['recall']:.3f} F1={cf_youden['f1']:.3f} | "
                            f"YoudenJ={cf_youden['youdenJ']:.3f}"
                        )
                        # Log OOD metrics to W&B
                        if self.args['use_wandb']:
                            wandb.log({
                                f"Task/{method_name}_auroc": metrics['auroc'],
                                f"Task/{method_name}_fpr95": metrics['fpr95'],
                                f"Task/{method_name}_aupr":  metrics['aupr_id'],
                                f"Task/{method_name}_youdenJ": metrics['youdenJ'],
                                f"Task/{method_name}_cf95_tp":  cf['tp'],
                                f"Task/{method_name}_cf95_fp":  cf['fp'],
                                f"Task/{method_name}_cf95_tn":  cf['tn'],
                                f"Task/{method_name}_cf95_fn":  cf['fn'],
                                f"Task/{method_name}_cf95_prec":  cf['precision'],
                                f"Task/{method_name}_cf95_rec":   cf['recall'],
                                f"Task/{method_name}_cf95_f1":    cf['f1'],
                                f"Task/{method_name}_cfJ_tp":  cf_youden['tp'],
                                f"Task/{method_name}_cfJ_fp":  cf_youden['fp'],
                                f"Task/{method_name}_cfJ_tn":  cf_youden['tn'],
                                f"Task/{method_name}_cfJ_fn":  cf_youden['fn'],
                                f"Task/{method_name}_cfJ_prec":  cf_youden['precision'],
                                f"Task/{method_name}_cfJ_rec":   cf_youden['recall'],
                                f"Task/{method_name}_cfJ_f1":    cf_youden['f1'],
                            })
                    else:
                        logging.error(f"{method_name}: Error - {metrics['error']}")
                        
                except Exception as e:
                    logging.error(f"{method_name} evaluation failed: {e}")
                    ood_results[method_name] = {'error': str(e), 'method': method_name}
            
        # Store results for trainer access
        self.latest_ood_results = ood_results
        self.latest_cl_results = {'cnn': cnn_accy, 'nme': nme_accy}
        
        # Store data for external visualization (will be used by trainer)
        self._visualization_data = {
            'id_features': id_features,
            'id_labels': id_labels,
            'ood_features': ood_features if self.ood_test_loader is not None else None,
            'score_distributions': score_distributions
        }
        '''
        {'MSP': {'method': 'MSP', 'auroc': 78.30048955815828, 'fpr95': 70.8080808080808, 'id_samples': 326, 'ood_samples': 990}, 'ODIN': {'method': 'ODIN', 'auroc': 81.93189564355208, 'fpr95': 62.62626262626263, 'id_samples': 326, 'ood_samples': 990}, 'Energy': {'method': 'Energy', 'auroc': 80.52023300489557, 'fpr95': 66.86868686868686, 'id_samples': 326, 'ood_samples': 990}}
        '''
        return ood_results, {'cnn': cnn_accy, 'nme': nme_accy}, score_distributions
    
    def clear_cached_data(self):
        """Clear cached data to free memory after T-SNE visualization"""
        if hasattr(self, '_cached_id_data'):
            del self._cached_id_data
        if hasattr(self, '_cached_ood_data'):
            del self._cached_ood_data
        logging.info("🧹 Cleared cached feature data to free memory")
    
    def _extract_data_batch(self, loader, extract_features=True, extract_logits=True):
        """
        통합된 데이터 추출 함수 - 한 번의 forward pass로 logits, features, labels 추출
        
        Args:
            loader: DataLoader
            extract_features: Whether to extract features for T-SNE
            extract_logits: Whether to extract logits for OOD detection
            
        Returns:
            dict: {'logits': tensor, 'features': array, 'labels': array}
        """
        self._network.eval()
        all_logits = []
        all_features = []
        all_labels = []
        C_t = self._classes_seen_so_far  # 누적 클래스 수

        with torch.no_grad():
            for _, inputs, targets in tqdm(loader, desc="Extracting data", leave=False):
                if isinstance(inputs, dict):
                    for m in inputs:
                        inputs[m] = inputs[m].to(self._device)
                else:
                    inputs = inputs.to(self._device)

                # 단일 forward pass로 모든 데이터 추출
                try:
                    # 네트워크 타입에 따른 조건부 호출 (TSN vs TBN 호환성)
                    if hasattr(self._network, 'forward') and 'mode' in self._network.forward.__code__.co_varnames:
                        # TSN 계열: mode 파라미터 지원
                        outputs = self._network(inputs, cur_task_size=C_t, mode='test')
                    else:
                        # TBN 계열: mode 파라미터 미지원, 기본 forward 사용
                        outputs = self._network(inputs)
                    
                    # Extract logits
                    if extract_logits:
                        all_logits.append(outputs["logits"].cpu())
                    
                    # Extract features  
                    if extract_features:
                        features = None
                        
                        # TBN과 TSN 호환 feature 추출
                        if hasattr(self._network, 'extract_vector'):
                            # TBN/TSN 공통: extract_vector 메서드 사용 (가장 안전)
                            features = self._network.extract_vector(inputs)
                        elif 'fusion_features' in outputs:
                            # TSN: fusion_features 사용 (이미 fusion된 feature)
                            features = outputs['fusion_features']
                        elif 'features' in outputs:
                            # TBN: features는 이미 fusion된 tensor
                            # TSN: features는 raw features (하지만 fusion_features 우선 사용됨)
                            features = outputs['features']
                        
                        # Ensure features are 2D [batch_size, feature_dim]
                        if features is not None:
                            if features.dim() > 2:
                                features = features.view(features.size(0), -1)
                            all_features.append(features.cpu())
                        else:
                            # Skip batch if features extraction failed
                            logging.warning(f"Features extraction failed for batch, skipping...")
                            continue
                    
                    all_labels.append(targets.cpu())
                    
                except Exception as e:
                    logging.warning(f"Data extraction failed for batch: {e}")
                    logging.warning(f"Batch targets shape: {targets.shape}, inputs type: {type(inputs)}")
                    logging.warning(f"Skipping this batch to avoid dummy data contamination")
                    # Skip failed batches completely instead of adding dummy data
                    # This prevents feature/label length mismatch and data contamination
                    continue

        # Prepare return dictionary
        result = {}
        
        if extract_logits and all_logits:
            result['logits'] = torch.cat(all_logits, dim=0)
            
        if extract_features and all_features:
            result['features'] = torch.cat(all_features, dim=0).numpy()
        else:
            result['features'] = None
            
        if all_labels:
            result['labels'] = torch.cat(all_labels, dim=0).numpy()
        else:
            result['labels'] = None
            
        logging.info(f"✅ Extracted data - Logits: {result['logits'].shape if 'logits' in result else 'None'}, "
                    f"Features: {result['features'].shape if result['features'] is not None else 'None'}, "
                    f"Labels: {result['labels'].shape if result['labels'] is not None else 'None'}")
        
        return result
    
    # Legacy wrapper functions for backward compatibility
    def _extract_logits_batch(self, loader):
        """Legacy function - extracts only logits"""
        result = self._extract_data_batch(loader, extract_features=False, extract_logits=True)
        return result.get('logits', torch.empty(0))
    
    def _extract_features_batch(self, loader):
        """Legacy function - extracts only features and labels"""
        result = self._extract_data_batch(loader, extract_features=True, extract_logits=False)
        return result.get('features'), result.get('labels')
  