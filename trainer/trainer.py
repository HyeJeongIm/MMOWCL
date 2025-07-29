import os
import sys
import copy
import logging
import datetime

from models.model_factory import get_model
from utils.utils import set_random_seed, set_device
from dataloader.data_manager import MyDataManager
from utils.result_analyzer import SimpleResultCollector

def train(args):
    """Main training function with experiment setup"""
    global experiment_dir, weights_dir

    # Create experiment directory
    lr_str = '_'.join([str(int(s)) for s in args["lr_steps"]])
    modality_str = ''.join(args["modality"]).lower()
    suffix = args.get("experiment_suffix", "")
    experiment_name = f"{args['dataset']}_{args['arch']}_{modality_str}_" \
                      f"lr{args['lr']}_lrst{lr_str}_dr{args['dropout']}_" \
                      f"ep{args['epochs']}_segs{args['num_segments']}_{suffix}"

    timestamp = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    experiment_dir = os.path.join(experiment_name, timestamp)
    weights_dir = os.path.join("weights", experiment_dir)
    os.makedirs(weights_dir, exist_ok=True)

    print(f"✓ Experiment directory created: {experiment_dir}")
    print(f"✓ Model weights will be saved to: {weights_dir}")
    
    # Run training for each seed
    seeds = args["seed"] if isinstance(args["seed"], list) else [args["seed"]]
    device = copy.deepcopy(args["device"])
    
    for seed in seeds:
        args["seed"] = seed
        args["device"] = device
        _train(args)
        
        

def _train(args):
    """Core OWCL training loop"""
    
    # Setup log directory
    log_dir = os.path.join("logs", experiment_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Define log filename
    log_name = f"{args['prefix']}_{args['seed']}_{args['model_name']}_" \
               f"{args['dataset']}_{args['init_cls']}_{args['increment']}.log"
    log_path = os.path.join(log_dir, log_name)

    # Configure logger (file + console)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ]
    )
        
    # Set random seed and device
    set_random_seed(args["seed"])
    args["device"] = set_device(args["device"])

    # Save config to file
    with open(os.path.join(log_dir, "args.txt"), "w") as f:
        f.write(str(args))
  
    
    # Initialize multi-modal OWCL model
    model = get_model(args["model_name"], args)
    
    # True
    # Freeze batch normalisation layers except the first
    if args["partialbn"]:
        model._network.backbone.freeze_fn('partialbn_parameters')
        
    image_tmpl = {}
    for m in args["modality"]:
        # Prepare dictionaries containing image name templates for each modality
        if m in ['RGB', 'RGBDiff']:
            image_tmpl[m] = "{:06d}.jpg"
        elif m == 'Flow':
            image_tmpl[m] = args["flow_prefix"] + "{}_{:06d}.jpg"
            
    data_manager = MyDataManager(model, image_tmpl, args)

    
    # Initialize result storage
    all_ood_results = {}  # Store OOD results for each task
    all_cl_results = {}   # Store continual learning results for each task
    
    # Initialize result collector
    collector = SimpleResultCollector(log_dir)
    collector.set_experiment_info(args)
    
    # Main OWCL training loop
    for task_id in range(data_manager.nb_tasks):
        print(f"\nTask {task_id + 1}/{data_manager.nb_tasks} Started")
        
        # Phase 1: Incremental training
        model.incremental_train(data_manager)
        
        # Phase 2: Evaluation
        ood_results, cl_results = model.evaluate_cl_ood()
        # Collect task results
        task_info = {
            'learning_classes': f"{model._known_classes}-{model._total_classes-1}",
            'ood_classes': f"{model._total_classes}-31",
            'train_samples': len(model.train_loader.dataset),
            'id_test_samples': len(model.test_loader.dataset),
            'ood_test_samples': len(model.ood_test_loader.dataset) if model.ood_test_loader else 0,
            'cl_accuracy': cl_results['cnn']['top1'],
            'ood_results': ood_results
        }
        collector.add_task_result(task_id, task_info)
        
        # Task Summary
        cl_acc = cl_results['cnn']['top1']
        ood_summary = []
        for method, metrics in ood_results.items():
            if 'error' not in metrics:
                ood_summary.append(f"{method}: {metrics['auroc']:.1f}%")
        
        ood_str = ", ".join(ood_summary) if ood_summary else "No OOD results"
        print(f"Task {task_id + 1} Completed - CL: {cl_acc:.1f}%, {ood_str}")
        
        # Phase 3: Update model state
        model.after_task()

        
        # Save checkpoint
        try:
            model.save_checkpoint(weights_dir, f"task_{task_id}_checkpoint")
        except AttributeError:
            pass  # Checkpoint saving not implemented
        
    # Save results and create visualizations
    print(f"\nSaving analysis results to: {log_dir}")
    json_path, csv_path = collector.save_results()
    print(f"✓ Results saved: {json_path}")
    print(f"✓ Summary saved: {csv_path}")
    
    print("Creating visualizations...")
    collector.create_visualizations()
    print(f"✓ Visualizations saved to: {collector.vis_dir}")
    
    # Final summary
    _log_final_summary(all_cl_results, all_ood_results, data_manager.nb_tasks)
    import ipdb; ipdb.set_trace()
    
def _log_final_summary(cl_results, ood_results, nb_tasks):
    """Log comprehensive final results summary"""
    logging.info(f"\n{'='*60}")
    logging.info("FINAL RESULTS SUMMARY")
    logging.info(f"{'='*60}")
    
    # Continual Learning Performance Summary
    logging.info("CONTINUAL LEARNING PERFORMANCE:")
    avg_accuracy = 0
    for task_id in range(nb_tasks):
        task_key = f"task_{task_id}"
        if task_key in cl_results:
            acc = cl_results[task_key]['cnn']['top1']
            avg_accuracy += acc
            logging.info(f"  Task {task_id + 1}: {acc:.2f}%")
    
    avg_accuracy /= nb_tasks
    logging.info(f"  Average Accuracy: {avg_accuracy:.2f}%")
    
    # OOD Detection Performance Summary
    logging.info("\nOOD DETECTION PERFORMANCE:")
    if ood_results:
        # Get OOD methods from first task
        first_task = list(ood_results.keys())[0]
        methods = list(ood_results[first_task].keys())
        
        for method in methods:
            avg_auroc = 0
            avg_fpr95 = 0
            valid_tasks = 0
            
            for task_id in range(nb_tasks):
                task_key = f"task_{task_id}"
                if task_key in ood_results and method in ood_results[task_key]:
                    metrics = ood_results[task_key][method]
                    if 'error' not in metrics:
                        avg_auroc += metrics['auroc']
                        avg_fpr95 += metrics['fpr95']
                        valid_tasks += 1
            
            if valid_tasks > 0:
                avg_auroc /= valid_tasks
                avg_fpr95 /= valid_tasks
                logging.info(f"  {method}: AUROC={avg_auroc:.1f}%, FPR95={avg_fpr95:.1f}%")
    
    logging.info(f"{'='*60}")