import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import itertools


class SimpleResultCollector:
    """Simple result collector for OWCL experiments"""
    
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.results = {
            'experiment_info': {},
            'tasks': []
        }
        
        # Create visualization directory
        self.vis_dir = os.path.join(log_dir, 'visualizations')
        os.makedirs(self.vis_dir, exist_ok=True)
    
    def set_experiment_info(self, args):
        """Set basic experiment information"""
        self.results['experiment_info'] = {
            'dataset': args['dataset'],
            'model': args['model_name'],
            'modalities': args['modality'],
            'init_cls': args['init_cls'],
            'increment': args['increment'],
            'ood_methods': args.get('ood_methods', []),
            'timestamp': datetime.now().isoformat()
        }
    
    def add_task_result(self, task_id, task_info):
        """Add task result to collection"""
        self.results['tasks'].append({
            'task_id': task_id + 1,
            'learning_classes': task_info['learning_classes'],
            'ood_classes': task_info['ood_classes'],
            'train_samples': task_info['train_samples'],
            'id_test_samples': task_info['id_test_samples'],
            'ood_test_samples': task_info['ood_test_samples'],
            'cl_accuracy': task_info['cl_accuracy'],
            'ood_results': task_info['ood_results'],
            'score_distributions': task_info.get('score_distributions', {})  # New: store ID/OOD scores
        })
    
    def save_results(self):
        """Save results to JSON and CSV"""
        # Save JSON
        json_path = os.path.join(self.log_dir, 'experiment_results.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save CSV
        csv_data = []
        for task in self.results['tasks']:
            row = {
                'Task': task['task_id'],
                'Learning_Classes': task['learning_classes'],
                'OOD_Classes': task['ood_classes'] if task['ood_test_samples'] > 0 else 'None',
                'Train_Samples': task['train_samples'],
                'ID_Test_Samples': task['id_test_samples'],
                'OOD_Test_Samples': task['ood_test_samples'],
                'CL_Accuracy': task['cl_accuracy']
            }
            
            # Add OOD results
            for method, metrics in task['ood_results'].items():
                if 'error' not in metrics:
                    row[f'{method}_AUROC'] = metrics.get('auroc', 0)
                    row[f'{method}_FPR95'] = metrics.get('fpr95', 0)
                    row[f'{method}_AUPR'] = metrics.get('aupr_id', 0)
                else:
                    row[f'{method}_AUROC'] = 0
                    row[f'{method}_FPR95'] = 100
                    row[f'{method}_AUPR'] = 0
            
            csv_data.append(row)
        
        csv_path = os.path.join(self.log_dir, 'task_summary.csv')
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        
        return json_path, csv_path
    
    def create_visualizations(self):
        """Create all visualizations"""
        csv_path = os.path.join(self.log_dir, 'task_summary.csv')
        if not os.path.exists(csv_path):
            print("No CSV file found. Run save_results() first.")
            return
        
        df = pd.read_csv(csv_path)
        
        # 1. Task Information Summary
        self._create_task_info_summary(df)
        
        # 2. CL Performance Analysis
        self._create_cl_performance(df)
        
        # 3. OOD Performance Visualization (AUROC + FPR95 charts)
        self._create_ood_performance_visualization(df)
        
        # 4. OOD Performance Tables (AUROC + FPR95 numerical tables)
        self._create_ood_performance_tables(df)
        
        # 5. Score Distribution Analysis (NEW)
        self._create_score_distributions()
        
        # 6. First Class (originally 26) vs OOD Analysis (NEW)  
        self._create_class26_analysis()
        
        self._create_confusion_heatmaps()   # ← NEW

    
    def _create_task_info_summary(self, df):
        """Create task information and sample distribution summary"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Task Configuration and Sample Distribution', fontsize=16, fontweight='bold')
        
        # 1. Task Info Table
        ax1.axis('tight')
        ax1.axis('off')
        table_data = df[['Task', 'Learning_Classes', 'OOD_Classes', 
                        'Train_Samples', 'ID_Test_Samples', 'OOD_Test_Samples']].values
        
        table = ax1.table(cellText=table_data,
                         colLabels=['Task', 'ID Classes', 'OOD Classes', 
                                   'Train', 'ID Test', 'OOD Test'],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Color coding for better readability
        for i in range(len(table_data)):
            for j in range(len(table_data[i])):
                if j == 2 and table_data[i][j] == 'None':  # OOD Classes column
                    table[(i+1, j)].set_facecolor('#ffcccc')  # Light red for no OOD
                elif j >= 3:  # Sample count columns
                    table[(i+1, j)].set_facecolor('#e6f3ff')  # Light blue
        
        ax1.set_title('Task Configuration Details', fontweight='bold')
        
        # 2. Sample Distribution Stacked Bar
        sample_data = df[['Train_Samples', 'ID_Test_Samples', 'OOD_Test_Samples']].values
        
        x = range(len(df))
        colors = ['#3498db', '#2ecc71', '#e74c3c']  # Blue, Green, Red
        labels = ['Train Samples', 'ID Test Samples', 'OOD Test Samples']
        
        bottom = np.zeros(len(df))
        for i, (samples, color, label) in enumerate(zip(sample_data.T, colors, labels)):
            bars = ax2.bar(x, samples, bottom=bottom, color=color, alpha=0.8, label=label)
            bottom += samples
            
            # Add value labels on bars
            for j, (bar, value) in enumerate(zip(bars, samples)):
                if value > 0:  # Only show label if value is not 0
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., 
                            bottom[j] - height/2, f'{int(value)}',
                            ha='center', va='center', fontweight='bold')
        
        ax2.set_xlabel('Task')
        ax2.set_ylabel('Number of Samples')
        ax2.set_title('Sample Distribution per Task', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'Task {i+1}' for i in range(len(df))])
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = os.path.join(self.vis_dir, '01_task_configuration_and_samples.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Task configuration saved: {save_path}")
    
    def _create_cl_performance(self, df):
        """Create continual learning performance analysis"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        fig.suptitle('Continual Learning Performance Analysis', fontsize=16, fontweight='bold')
        
        # CL Accuracy Trend
        ax.plot(df['Task'], df['CL_Accuracy'], marker='o', linewidth=3,
                markersize=10, color='#e74c3c', markerfacecolor='white',
                markeredgewidth=2, label='CL Accuracy')
        ax.fill_between(df['Task'], df['CL_Accuracy'], alpha=0.3, color='#e74c3c')
        
        # Add value labels
        for i, (task, acc) in enumerate(zip(df['Task'], df['CL_Accuracy'])):
            ax.annotate(f'{acc:.1f}%', (task, acc), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        ax.set_xlabel('Task Number')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('CL Accuracy Trend Across Tasks', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        save_path = os.path.join(self.vis_dir, '02_continual_learning_performance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ CL performance saved: {save_path}")
    
    def _create_ood_performance_visualization(self, df):
        """Create OOD performance visualization (AUROC + FPR95 + AUPR charts)"""
        ood_auroc_cols = [col for col in df.columns if col.endswith('_AUROC')]
        ood_fpr_cols = [col for col in df.columns if col.endswith('_FPR95')]
        ood_aupr_cols = [col for col in df.columns if col.endswith('_AUPR')]
        
        if not ood_auroc_cols:
            return
        
        # Filter out tasks with no OOD samples
        df_with_ood = df[df['OOD_Test_Samples'] > 0].copy()
        if len(df_with_ood) == 0:
            return
        
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('OOD Detection Performance Visualization (AUROC + FPR95 + AUPR)', fontsize=16, fontweight='bold')
        
        methods = [col.replace('_AUROC', '') for col in ood_auroc_cols]
        colors = ['#3498db', '#2ecc71', '#9b59b6', '#e67e22']
        
        # 1. AUROC Bar Chart
        x = np.arange(len(df_with_ood))
        width = 0.8 / len(methods)
        
        has_auroc_plots = False
        for i, (method, color) in enumerate(zip(methods, colors)):
            values = df_with_ood[f'{method}_AUROC'].values
            bars = ax1.bar(x + i * width, values, width, label=method, 
                          color=color, alpha=0.8)
            has_auroc_plots = True
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax1.set_xlabel('Task Number')
        ax1.set_ylabel('AUROC (%)')
        ax1.set_title('AUROC by Task and Method (Higher = Better)', fontweight='bold')
        ax1.set_xticks(x + width * (len(methods) - 1) / 2)
        ax1.set_xticklabels([f'Task {int(task)}' for task in df_with_ood['Task']])
        if has_auroc_plots:
            ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 100)
        
        # 2. AUROC Line Chart
        has_auroc_line_plots = False
        for i, col in enumerate(ood_auroc_cols):
            method_name = col.replace('_AUROC', '')
            ax2.plot(df_with_ood['Task'], df_with_ood[col], 
                    marker='o', linewidth=2.5, markersize=8,
                    color=colors[i % len(colors)], label=method_name)
            has_auroc_line_plots = True
        
        ax2.set_xlabel('Task Number')
        ax2.set_ylabel('AUROC (%)')
        ax2.set_title('AUROC Trends (Higher = Better)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        if has_auroc_line_plots:
            ax2.legend()
        ax2.set_ylim(0, 100)
        
        # 3. FPR95 Bar Chart
        has_fpr_plots = False
        for i, (method, color) in enumerate(zip(methods, colors)):
            fpr_col = f'{method}_FPR95'
            if fpr_col in df_with_ood.columns:
                values = df_with_ood[fpr_col].values
                bars = ax3.bar(x + i * width, values, width, label=method, 
                              color=color, alpha=0.8)
                has_fpr_plots = True
                
                # Add value labels
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax3.set_xlabel('Task Number')
        ax3.set_ylabel('FPR95 (%)')
        ax3.set_title('FPR95 by Task and Method (Lower = Better)', fontweight='bold')
        ax3.set_xticks(x + width * (len(methods) - 1) / 2)
        ax3.set_xticklabels([f'Task {int(task)}' for task in df_with_ood['Task']])
        if has_fpr_plots:
            ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim(0, 100)
        
        # 4. FPR95 Line Chart
        has_fpr_line_plots = False
        for i, col in enumerate(ood_fpr_cols):
            method_name = col.replace('_FPR95', '')
            if col in df_with_ood.columns:
                ax4.plot(df_with_ood['Task'], df_with_ood[col], 
                        marker='s', linewidth=2.5, markersize=8,
                        color=colors[i % len(colors)], label=method_name)
                has_fpr_line_plots = True
        
        ax4.set_xlabel('Task Number')
        ax4.set_ylabel('FPR95 (%)')
        ax4.set_title('FPR95 Trends (Lower = Better)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        if has_fpr_line_plots:
            ax4.legend()
        ax4.set_ylim(0, 100)
        
        # 5. AUPR Bar Chart
        has_aupr_plots = False
        for i, (method, color) in enumerate(zip(methods, colors)):
            aupr_col = f'{method}_AUPR'
            if aupr_col in df_with_ood.columns:
                values = df_with_ood[aupr_col].values
                bars = ax5.bar(x + i * width, values, width, label=method, 
                              color=color, alpha=0.8)
                has_aupr_plots = True
                
                # Add value labels
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax5.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax5.set_xlabel('Task Number')
        ax5.set_ylabel('AUPR (%)')
        ax5.set_title('AUPR by Task and Method (Higher = Better)', fontweight='bold')
        ax5.set_xticks(x + width * (len(methods) - 1) / 2)
        ax5.set_xticklabels([f'Task {int(task)}' for task in df_with_ood['Task']])
        if has_aupr_plots:
            ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.set_ylim(0, 100)
        
        # 6. AUPR Line Chart
        has_aupr_line_plots = False
        for i, col in enumerate(ood_aupr_cols):
            method_name = col.replace('_AUPR', '')
            if col in df_with_ood.columns:
                ax6.plot(df_with_ood['Task'], df_with_ood[col], 
                        marker='^', linewidth=2.5, markersize=8,
                        color=colors[i % len(colors)], label=method_name)
                has_aupr_line_plots = True
        
        ax6.set_xlabel('Task Number')
        ax6.set_ylabel('AUPR (%)')
        ax6.set_title('AUPR Trends (Higher = Better)', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        if has_aupr_line_plots:
            ax6.legend()
        ax6.set_ylim(0, 100)
        
        plt.tight_layout()
        save_path = os.path.join(self.vis_dir, '03_ood_performance_visualization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ OOD performance visualization (AUROC + FPR95 + AUPR) saved: {save_path}")
    
    def _create_ood_performance_tables(self, df):
        """Create OOD performance tables (AUROC + FPR95 + AUPR numerical tables)"""
        ood_auroc_cols = [col for col in df.columns if col.endswith('_AUROC')]
        ood_fpr_cols = [col for col in df.columns if col.endswith('_FPR95')]
        ood_aupr_cols = [col for col in df.columns if col.endswith('_AUPR')]
        
        if not ood_auroc_cols:
            return
        
        # Filter out tasks with no OOD samples
        df_with_ood = df[df['OOD_Test_Samples'] > 0].copy()
        if len(df_with_ood) == 0:
            return
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle('OOD Detection Performance Tables (AUROC + FPR95 + AUPR)', fontsize=16, fontweight='bold')
        
        methods = [col.replace('_AUROC', '') for col in ood_auroc_cols]
        
        # 1. AUROC Table
        ax1.axis('tight')
        ax1.axis('off')
        
        # Prepare AUROC table data
        auroc_table_data = []
        for _, row in df_with_ood.iterrows():
            row_data = [f"Task {int(row['Task'])}"]
            for method in methods:
                auroc_value = row[f'{method}_AUROC']
                row_data.append(f"{auroc_value:.1f}%")
            auroc_table_data.append(row_data)
        
        # Add average row
        avg_row = ["Average"]
        for method in methods:
            avg_auroc = df_with_ood[f'{method}_AUROC'].mean()
            avg_row.append(f"{avg_auroc:.1f}%")
        auroc_table_data.append(avg_row)
        
        # Add best row
        best_row = ["Best"]
        for method in methods:
            best_auroc = df_with_ood[f'{method}_AUROC'].max()
            best_row.append(f"{best_auroc:.1f}%")
        auroc_table_data.append(best_row)
        
        # Create AUROC table
        auroc_table = ax1.table(cellText=auroc_table_data,
                               colLabels=['Task'] + [f'{method}\nAUROC' for method in methods],
                               cellLoc='center', loc='center')
        auroc_table.auto_set_font_size(False)
        auroc_table.set_fontsize(11)
        auroc_table.scale(1, 2.5)
        
        # Color coding for AUROC table
        for i in range(len(auroc_table_data)):
            if i == len(auroc_table_data) - 2:  # Average row
                for j in range(len(methods) + 1):
                    auroc_table[(i+1, j)].set_facecolor('#e6f3ff')  # Light blue
            elif i == len(auroc_table_data) - 1:  # Best row
                for j in range(len(methods) + 1):
                    auroc_table[(i+1, j)].set_facecolor('#e6ffe6')  # Light green
            else:
                # Regular task rows
                for j in range(1, len(methods) + 1):
                    auroc_table[(i+1, j)].set_facecolor('#f8f9fa')  # Very light gray
        
        ax1.set_title('AUROC Performance Table\n(Higher = Better)', fontweight='bold', pad=20)
        
        # 2. FPR95 Table
        ax2.axis('tight')
        ax2.axis('off')
        
        # Prepare FPR95 table data
        fpr_table_data = []
        valid_fpr_methods = []
        
        # Check which methods have FPR95 data
        for method in methods:
            if f'{method}_FPR95' in df_with_ood.columns:
                valid_fpr_methods.append(method)
        
        if valid_fpr_methods:
            for _, row in df_with_ood.iterrows():
                row_data = [f"Task {int(row['Task'])}"]
                for method in valid_fpr_methods:
                    fpr_value = row[f'{method}_FPR95']
                    row_data.append(f"{fpr_value:.1f}%")
                fpr_table_data.append(row_data)
            
            # Add average row
            avg_row = ["Average"]
            for method in valid_fpr_methods:
                avg_fpr = df_with_ood[f'{method}_FPR95'].mean()
                avg_row.append(f"{avg_fpr:.1f}%")
            fpr_table_data.append(avg_row)
            
            # Add best row (minimum for FPR95)
            best_row = ["Best"]
            for method in valid_fpr_methods:
                best_fpr = df_with_ood[f'{method}_FPR95'].min()
                best_row.append(f"{best_fpr:.1f}%")
            fpr_table_data.append(best_row)
            
            # Create FPR95 table
            fpr_table = ax2.table(cellText=fpr_table_data,
                                 colLabels=['Task'] + [f'{method}\nFPR95' for method in valid_fpr_methods],
                                 cellLoc='center', loc='center')
            fpr_table.auto_set_font_size(False)
            fpr_table.set_fontsize(11)
            fpr_table.scale(1, 2.5)
            
            # Color coding for FPR95 table
            for i in range(len(fpr_table_data)):
                if i == len(fpr_table_data) - 2:  # Average row
                    for j in range(len(valid_fpr_methods) + 1):
                        fpr_table[(i+1, j)].set_facecolor('#ffe6e6')  # Light red
                elif i == len(fpr_table_data) - 1:  # Best row
                    for j in range(len(valid_fpr_methods) + 1):
                        fpr_table[(i+1, j)].set_facecolor('#e6ffe6')  # Light green
                else:
                    # Regular task rows
                    for j in range(1, len(valid_fpr_methods) + 1):
                        fpr_table[(i+1, j)].set_facecolor('#f8f9fa')  # Very light gray
        
        ax2.set_title('FPR95 Performance Table\n(Lower = Better)', fontweight='bold', pad=20)
        
        # 3. AUPR Table
        ax3.axis('tight')
        ax3.axis('off')
        
        # Prepare AUPR table data
        aupr_table_data = []
        valid_aupr_methods = []
        
        # Check which methods have AUPR data
        for method in methods:
            if f'{method}_AUPR' in df_with_ood.columns:
                valid_aupr_methods.append(method)
        
        if valid_aupr_methods:
            for _, row in df_with_ood.iterrows():
                row_data = [f"Task {int(row['Task'])}"]
                for method in valid_aupr_methods:
                    aupr_value = row[f'{method}_AUPR']
                    row_data.append(f"{aupr_value:.1f}%")
                aupr_table_data.append(row_data)
            
            # Add average row
            avg_row = ["Average"]
            for method in valid_aupr_methods:
                avg_aupr = df_with_ood[f'{method}_AUPR'].mean()
                avg_row.append(f"{avg_aupr:.1f}%")
            aupr_table_data.append(avg_row)
            
            # Add best row (maximum for AUPR)
            best_row = ["Best"]
            for method in valid_aupr_methods:
                best_aupr = df_with_ood[f'{method}_AUPR'].max()
                best_row.append(f"{best_aupr:.1f}%")
            aupr_table_data.append(best_row)
            
            # Create AUPR table
            aupr_table = ax3.table(cellText=aupr_table_data,
                                 colLabels=['Task'] + [f'{method}\nAUPR' for method in valid_aupr_methods],
                                 cellLoc='center', loc='center')
            aupr_table.auto_set_font_size(False)
            aupr_table.set_fontsize(11)
            aupr_table.scale(1, 2.5)
            
            # Color coding for AUPR table
            for i in range(len(aupr_table_data)):
                if i == len(aupr_table_data) - 2:  # Average row
                    for j in range(len(valid_aupr_methods) + 1):
                        aupr_table[(i+1, j)].set_facecolor('#e6f3ff')  # Light blue
                elif i == len(aupr_table_data) - 1:  # Best row
                    for j in range(len(valid_aupr_methods) + 1):
                        aupr_table[(i+1, j)].set_facecolor('#e6ffe6')  # Light green
                else:
                    # Regular task rows
                    for j in range(1, len(valid_aupr_methods) + 1):
                        aupr_table[(i+1, j)].set_facecolor('#f8f9fa')  # Very light gray
        
        ax3.set_title('AUPR Performance Table\n(Higher = Better)', fontweight='bold', pad=20)
        
        plt.tight_layout()
        save_path = os.path.join(self.vis_dir, '04_ood_performance_tables.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ OOD performance tables (AUROC + FPR95 + AUPR) saved: {save_path}")
    
    def _create_score_distributions(self):
        """Create ID/OOD score distribution analysis for each task"""
        print("Creating score distribution analysis...")
        
        # Create subdirectory for score distributions
        score_dir = os.path.join(self.vis_dir, 'score_distributions')
        os.makedirs(score_dir, exist_ok=True)
        
        # Collect all available methods and tasks
        all_methods = set()
        tasks_with_scores = []
        
        for task in self.results['tasks']:
            if 'score_distributions' in task and task['score_distributions'] and task['ood_test_samples'] > 0:
                tasks_with_scores.append(task)
                all_methods.update(task['score_distributions'].keys())
        
        if not tasks_with_scores or not all_methods:
            print("No score distribution data available.")
            return
        
        # Beautiful color palettes
        task_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
        id_color = '#2E86C1'    # Beautiful blue
        ood_color = '#E74C3C'   # Beautiful red
        
        # Create visualization for each method
        for method in sorted(all_methods):
            print(f"Creating {method} score distribution visualization...")
            
            # Collect data for this method across all tasks
            method_data = []
            for task in tasks_with_scores:
                if method in task['score_distributions']:
                    score_data = task['score_distributions'][method]
                    id_scores = np.array(score_data.get('id_scores', []))
                    ood_scores = np.array(score_data.get('ood_scores', []))
                    
                    if len(id_scores) > 0 and len(ood_scores) > 0:
                        method_data.append({
                            'task_id': task['task_id'],
                            'id_scores': id_scores,
                            'ood_scores': ood_scores,
                            'learning_classes': task['learning_classes'],
                            'ood_classes': task['ood_classes']
                        })
            
            if not method_data:
                continue
            
            # Create figure for this method
            n_tasks = len(method_data)
            fig, axes = plt.subplots(1, n_tasks, figsize=(6*n_tasks, 8))
            if n_tasks == 1:
                axes = [axes]
            
            fig.suptitle(f'{method} Score Distributions Across Tasks', fontsize=18, fontweight='bold', y=0.95)
            
            # Global min/max for consistent y-axis
            all_scores = []
            for data in method_data:
                all_scores.extend(data['id_scores'])
                all_scores.extend(data['ood_scores'])
            global_min, global_max = np.min(all_scores), np.max(all_scores)
            
            for idx, data in enumerate(method_data):
                ax = axes[idx]
                task_id = data['task_id']
                id_scores = data['id_scores']
                ood_scores = data['ood_scores']
                
                # Calculate bins for consistent visualization
                bins = np.linspace(global_min, global_max, 40)
                
                # Plot histograms with beautiful colors
                ax.hist(id_scores, bins=bins, alpha=0.75, color=id_color, 
                       label=f'ID (n={len(id_scores)})', density=True, 
                       edgecolor='white', linewidth=1.2)
                ax.hist(ood_scores, bins=bins, alpha=0.75, color=ood_color, 
                       label=f'OOD (n={len(ood_scores)})', density=True, 
                       edgecolor='white', linewidth=1.2)
                
                # Add vertical lines for means with better styling
                id_mean = id_scores.mean()
                ood_mean = ood_scores.mean()
                
                ax.axvline(id_mean, color='#1B4F72', linestyle='--', linewidth=2.5, 
                          label=f'ID μ: {id_mean:.3f}', alpha=0.8)
                ax.axvline(ood_mean, color='#922B21', linestyle='--', linewidth=2.5, 
                          label=f'OOD μ: {ood_mean:.3f}', alpha=0.8)
                
                # Calculate and display overlap
                overlap_area = self._calculate_overlap(id_scores, ood_scores, bins)
                
                # Beautiful styling
                ax.set_title(f'Task {task_id}\nID: {data["learning_classes"]} | OOD: {data["ood_classes"]}\nOverlap: {overlap_area:.1f}%', 
                           fontweight='bold', fontsize=12, pad=15)
                ax.set_xlabel('Score', fontsize=11, fontweight='bold')
                if idx == 0:
                    ax.set_ylabel('Density', fontsize=11, fontweight='bold')
                
                # Legend with better positioning
                ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, 
                         fontsize=9, framealpha=0.9)
                
                # Grid and styling
                ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
                ax.set_facecolor('#FAFAFA')
                
                # Set consistent y-axis limits
                ax.set_ylim(bottom=0)
                
                # Add task progression indicator with color
                task_color = task_colors[idx % len(task_colors)]
                ax.add_patch(plt.Rectangle((0.02, 0.95), 0.05, 0.03, 
                           transform=ax.transAxes, facecolor=task_color, 
                           alpha=0.8, clip_on=False))
                ax.text(0.1, 0.965, f'Task {task_id}', transform=ax.transAxes, 
                       fontsize=10, fontweight='bold', va='center',
                       color=task_color)
                
                # Add separability score
                avg_std = (id_scores.std() + ood_scores.std()) / 2
                separability = abs(id_mean - ood_mean) / (avg_std + 1e-8)
                ax.text(0.02, 0.85, f'Separability: {separability:.2f}', 
                       transform=ax.transAxes, fontsize=9, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            save_path = os.path.join(score_dir, f'{method.lower()}_score_distribution.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"✓ {method} score distribution saved: {save_path}")
        
        # Create summary comparison (keep the existing one but with better colors)
        self._create_score_summary_comparison(score_dir)
    
    def _calculate_overlap(self, id_scores, ood_scores, bins):
        """Calculate overlap percentage between ID and OOD score distributions"""
        id_hist, _ = np.histogram(id_scores, bins=bins, density=True)
        ood_hist, _ = np.histogram(ood_scores, bins=bins, density=True)
        
        # Normalize histograms to sum to 1
        id_hist = id_hist / np.sum(id_hist)
        ood_hist = ood_hist / np.sum(ood_hist)
        
        # Calculate overlap as minimum of the two distributions
        overlap = np.sum(np.minimum(id_hist, ood_hist)) * 100
        return overlap
    
    def _create_score_summary_comparison(self, score_dir):
        """Create summary comparison of score separability across all tasks"""
        tasks_with_scores = [task for task in self.results['tasks'] 
                           if task.get('score_distributions') and task['ood_test_samples'] > 0]
        
        if not tasks_with_scores:
            return
        
        # Collect data for summary
        summary_data = {}
        
        for task in tasks_with_scores:
            task_id = task['task_id']
            score_data = task['score_distributions']
            
            for method in score_data:
                if method not in summary_data:
                    summary_data[method] = {
                        'tasks': [],
                        'separability': [],
                        'id_means': [],
                        'ood_means': [],
                        'overlap_areas': []
                    }
                
                id_scores = np.array(score_data[method].get('id_scores', []))
                ood_scores = np.array(score_data[method].get('ood_scores', []))
                
                if len(id_scores) > 0 and len(ood_scores) > 0:
                    # Calculate separability (difference in means / average std)
                    id_mean, ood_mean = id_scores.mean(), ood_scores.mean()
                    avg_std = (id_scores.std() + ood_scores.std()) / 2
                    separability = abs(id_mean - ood_mean) / (avg_std + 1e-8)
                    
                    # Calculate overlap
                    all_scores = np.concatenate([id_scores, ood_scores])
                    bins = np.linspace(all_scores.min(), all_scores.max(), 50)
                    overlap = self._calculate_overlap(id_scores, ood_scores, bins)
                    
                    summary_data[method]['tasks'].append(task_id)
                    summary_data[method]['separability'].append(separability)
                    summary_data[method]['id_means'].append(id_mean)
                    summary_data[method]['ood_means'].append(ood_mean)
                    summary_data[method]['overlap_areas'].append(overlap)
        
        # Create summary plots with beautiful colors
        if summary_data:
            methods = list(summary_data.keys())
            n_methods = len(methods)
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Score Distribution Summary Across All Tasks', fontsize=18, fontweight='bold')
            
            # Beautiful color palette
            colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
            
            # 1. Separability across tasks
            ax1 = axes[0, 0]
            for i, method in enumerate(methods):
                tasks = summary_data[method]['tasks']
                separability = summary_data[method]['separability']
                ax1.plot(tasks, separability, marker='o', linewidth=3, markersize=8,
                        color=colors[i % len(colors)], label=method, alpha=0.8,
                        markerfacecolor='white', markeredgewidth=2)
            
            ax1.set_xlabel('Task Number', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Separability Score', fontsize=12, fontweight='bold')
            ax1.set_title('ID/OOD Separability Across Tasks\n(Higher = Better)', fontweight='bold', fontsize=14)
            ax1.legend(frameon=True, fancybox=True, shadow=True)
            ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
            ax1.set_facecolor('#FAFAFA')
            
            # 2. Overlap percentage across tasks
            ax2 = axes[0, 1]
            for i, method in enumerate(methods):
                tasks = summary_data[method]['tasks']
                overlap = summary_data[method]['overlap_areas']
                ax2.plot(tasks, overlap, marker='s', linewidth=3, markersize=8,
                        color=colors[i % len(colors)], label=method, alpha=0.8,
                        markerfacecolor='white', markeredgewidth=2)
            
            ax2.set_xlabel('Task Number', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Overlap Percentage (%)', fontsize=12, fontweight='bold')
            ax2.set_title('ID/OOD Distribution Overlap\n(Lower = Better)', fontweight='bold', fontsize=14)
            ax2.legend(frameon=True, fancybox=True, shadow=True)
            ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
            ax2.set_facecolor('#FAFAFA')
            
            # 3. Mean score trends
            ax3 = axes[1, 0]
            for i, method in enumerate(methods):
                tasks = summary_data[method]['tasks']
                id_means = summary_data[method]['id_means']
                ood_means = summary_data[method]['ood_means']
                
                ax3.plot(tasks, id_means, marker='o', linewidth=3, markersize=8, linestyle='-',
                        color=colors[i % len(colors)], label=f'{method} ID', alpha=0.8,
                        markerfacecolor='white', markeredgewidth=2)
                ax3.plot(tasks, ood_means, marker='s', linewidth=3, markersize=8, linestyle='--',
                        color=colors[i % len(colors)], label=f'{method} OOD', alpha=0.6,
                        markerfacecolor='white', markeredgewidth=2)
            
            ax3.set_xlabel('Task Number', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Mean Score', fontsize=12, fontweight='bold')
            ax3.set_title('Mean Score Trends', fontweight='bold', fontsize=14)
            ax3.legend(frameon=True, fancybox=True, shadow=True)
            ax3.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
            ax3.set_facecolor('#FAFAFA')
            
            # 4. Method comparison summary
            ax4 = axes[1, 1]
            method_avg_sep = [np.mean(summary_data[m]['separability']) for m in methods]
            method_avg_overlap = [np.mean(summary_data[m]['overlap_areas']) for m in methods]
            
            x = np.arange(len(methods))
            width = 0.35
            
            bars1 = ax4.bar(x - width/2, method_avg_sep, width, label='Avg Separability', 
                           color='#2ECC71', alpha=0.8, edgecolor='white', linewidth=1.5)
            ax4_twin = ax4.twinx()
            bars2 = ax4_twin.bar(x + width/2, method_avg_overlap, width, label='Avg Overlap (%)', 
                                color='#E74C3C', alpha=0.8, edgecolor='white', linewidth=1.5)
            
            ax4.set_xlabel('OOD Method', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Average Separability', color='#2ECC71', fontsize=12, fontweight='bold')
            ax4_twin.set_ylabel('Average Overlap (%)', color='#E74C3C', fontsize=12, fontweight='bold')
            ax4.set_title('Method Performance Summary', fontweight='bold', fontsize=14)
            ax4.set_xticks(x)
            ax4.set_xticklabels(methods)
            ax4.set_facecolor('#FAFAFA')
            
            # Add value labels with better styling
            for bar, val in zip(bars1, method_avg_sep):
                ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                        f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
            for bar, val in zip(bars2, method_avg_overlap):
                ax4_twin.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                             f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            ax4.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
            
            plt.tight_layout()
            save_path = os.path.join(score_dir, '00_score_summary_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"✓ Score summary comparison saved: {save_path}")
    
    def _create_class26_analysis(self):
        """Create First Class (originally 26) score analysis aligned with _create_score_distributions pattern."""
        print("Creating First Class (originally 26) analysis...")

        class26_dir = os.path.join(self.vis_dir, 'class26_analysis')
        os.makedirs(class26_dir, exist_ok=True)

        tasks = self.results.get('tasks', [])
        if not tasks:
            print("No tasks in results.")
            return

        # 1) 어떤 방법론(method)이든 한 번이라도 class_26_scores를 가진 적이 있으면 후보에 포함
        all_methods = set()
        for task in tasks:
            sd = task.get('score_distributions', {})
            for m, d in sd.items():
                if isinstance(d, dict) and len(d.get('class_26_scores', [])) > 0:
                    all_methods.add(m)

        if not all_methods:
            print("No Class 26 data available for analysis.")
            return

        # 2) task_id 기준 정렬된 task 리스트
        tasks_sorted = sorted(tasks, key=lambda t: t.get('task_id', 0))

        # 3) 시각화 (Class26 vs OOD / Class26 Solo)
        self._create_class26_vs_ood_multi_task(tasks_sorted, all_methods, class26_dir)
        self._create_class26_solo_multi_task(tasks_sorted, all_methods, class26_dir)

        # 4) 요약(진화) 그래프
        self._create_class26_evolution_summary(tasks_sorted, all_methods, class26_dir)
        print(f"✓ Class 26 analysis saved to: {class26_dir}")
    
    def _create_class26_vs_ood_multi_task(self, tasks_sorted, all_methods, class26_dir):
        """For each method, draw Class 26 vs OOD distributions across ALL tasks (one figure per method).
        - 각 task 열을 반드시 유지하고, 데이터 없으면 'No Class 26 data' 박스 출력
        """
        class26_color = '#2E86C1'
        ood_color = '#E74C3C'
        task_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']

        # 전역 min/max를 위해 미리 점수 모으기
        for method in sorted(all_methods):
            # task별 데이터 준비 (빈 task도 자리 유지)
            per_task = []
            global_vals = []

            for task in tasks_sorted:
                sd = task.get('score_distributions', {})
                d = sd.get(method, {})
                c26 = np.array(d.get('class_26_scores', []))
                ood = np.array(d.get('ood_scores', []))
                has_ood = len(ood) > 0

                if len(c26) > 0:
                    global_vals.extend(c26.tolist())
                if has_ood:
                    global_vals.extend(ood.tolist())

                per_task.append({
                    'task_id': task['task_id'],
                    'learning_classes': task.get('learning_classes'),
                    'ood_classes': task.get('ood_classes') if task.get('ood_test_samples', 0) > 0 else 'None',
                    'class_26_scores': c26,
                    'ood_scores': ood,
                    'has_ood': has_ood,
                    'class_26_count': int(d.get('class_26_count', len(c26)))
                })

            # 전역 스케일 계산 (전부 비어있으면 스킵)
            global_vals = np.array(global_vals)
            if global_vals.size == 0:
                # 모든 task가 비어있다면 이 method는 스킵
                print(f"[Class26 vs OOD] Skip {method}: no data across tasks.")
                continue
            gmin, gmax = float(global_vals.min()), float(global_vals.max())
            bins = np.linspace(gmin, gmax, 40)

            # 그림 생성: 열의 개수 = 전체 task 개수
            n_tasks = len(per_task)
            fig, axes = plt.subplots(1, n_tasks, figsize=(6*n_tasks, 8))
            if n_tasks == 1:
                axes = [axes]

            fig.suptitle(f'{method}: Class 26 vs OOD Score Distributions Across Tasks',
                         fontsize=18, fontweight='bold', y=0.95)

            for idx, data in enumerate(per_task):
                ax = axes[idx]
                ax.set_facecolor('#FAFAFA')
                color_tag = task_colors[idx % len(task_colors)]

                c26 = data['class_26_scores']
                ood = data['ood_scores']
                has_ood = data['has_ood']

                if c26.size > 0:
                    ax.hist(c26, bins=bins, alpha=0.75, color=class26_color,
                            label=f'Class 26 (n={data["class_26_count"]})', density=True,
                            edgecolor='white', linewidth=1.2)
                    c26_mean = float(c26.mean())
                    ax.axvline(c26_mean, color='#1B4F72', linestyle='--', linewidth=2.5,
                               label=f'Class 26 μ: {c26_mean:.3f}', alpha=0.8)
                else:
                    # 데이터가 없을 때 안내 박스
                    ax.text(0.5, 0.5, 'No Class 26 data',
                            ha='center', va='center', transform=ax.transAxes,
                            fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                    c26_mean = None

                if has_ood:
                    ax.hist(ood, bins=bins, alpha=0.75, color=ood_color,
                            label=f'OOD (n={len(ood)})', density=True,
                            edgecolor='white', linewidth=1.2)
                    ood_mean = float(ood.mean())
                    ax.axvline(ood_mean, color='#922B21', linestyle='--', linewidth=2.5,
                               label=f'OOD μ: {ood_mean:.3f}', alpha=0.8)
                else:
                    ood_mean = None

                # Overlap / Separability (둘 다 있을 때만)
                subtitle_extra = ''
                if c26.size > 0 and has_ood:
                    overlap = self._calculate_overlap(c26, ood, bins)
                    avg_std = (c26.std() + (ood.std() if has_ood else 0.0)) / 2.0 + 1e-8
                    sep = abs(c26_mean - ood_mean) / avg_std
                    subtitle_extra = f' • Overlap: {overlap:.1f}% • Sep: {sep:.2f}'

                ax.set_title(
                    f'Task {data["task_id"]}\nID: {data["learning_classes"]} | OOD: {data["ood_classes"]}{subtitle_extra}',
                    fontweight='bold', fontsize=12, pad=15
                )
                ax.set_xlabel('Score', fontsize=11, fontweight='bold')
                if idx == 0:
                    ax.set_ylabel('Density', fontsize=11, fontweight='bold')

                ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True,
                          fontsize=9, framealpha=0.9)
                ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
                ax.set_ylim(bottom=0)

                # Task 인디케이터
                ax.add_patch(plt.Rectangle((0.02, 0.95), 0.05, 0.03, transform=ax.transAxes,
                                           facecolor=color_tag, alpha=0.8, clip_on=False))
                ax.text(0.1, 0.965, f'Task {data["task_id"]}', transform=ax.transAxes,
                        fontsize=10, fontweight='bold', va='center', color=color_tag)

            plt.tight_layout()
            save_path = os.path.join(class26_dir, f'{method.lower()}_class26_vs_ood.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"✓ {method} Class 26 vs OOD distribution saved: {save_path}")
    
    def _create_class26_solo_multi_task(self, tasks_sorted, all_methods, class26_dir):
        """For each method, draw Class 26-only distributions across ALL tasks (one figure per method).
        - 데이터가 없어도 자리 유지 + 안내문 출력
        """
        task_colors = ['#2E86C1', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']

        for method in sorted(all_methods):
            per_task = []
            global_vals = []

            for task in tasks_sorted:
                sd = task.get('score_distributions', {})
                d = sd.get(method, {})
                c26 = np.array(d.get('class_26_scores', []))
                per_task.append({
                    'task_id': task['task_id'],
                    'learning_classes': task.get('learning_classes'),
                    'class_26_scores': c26,
                    'class_26_count': int(d.get('class_26_count', len(c26)))
                })
                if c26.size > 0:
                    global_vals.extend(c26.tolist())

            global_vals = np.array(global_vals)
            if global_vals.size == 0:
                print(f"[Class26 solo] Skip {method}: no class 26 scores across tasks.")
                continue
            gmin, gmax = float(global_vals.min()), float(global_vals.max())
            bins = np.linspace(gmin, gmax, 30)

            n_tasks = len(per_task)
            fig, axes = plt.subplots(1, n_tasks, figsize=(6*n_tasks, 8))
            if n_tasks == 1:
                axes = [axes]

            fig.suptitle(f'{method}: Class 26 Distribution Evolution Across Tasks',
                         fontsize=18, fontweight='bold', y=0.95)

            for idx, data in enumerate(per_task):
                ax = axes[idx]
                ax.set_facecolor('#FAFAFA')
                color = task_colors[idx % len(task_colors)]
                c26 = data['class_26_scores']

                if c26.size > 0:
                    n, bins_ret, patches = ax.hist(c26, bins=bins, alpha=0.8, color=color,
                                                   density=True, edgecolor='white', linewidth=1.2)
                    mu, sd = float(c26.mean()), float(c26.std())
                    ax.axvline(mu, color='darkred', linestyle='-', linewidth=3,
                               label=f'Mean: {mu:.3f}', alpha=0.9)
                    ax.axvline(mu - sd, color='darkred', linestyle='--', linewidth=2,
                               label=f'μ-σ: {mu - sd:.3f}', alpha=0.7)
                    ax.axvline(mu + sd, color='darkred', linestyle='--', linewidth=2,
                               label=f'μ+σ: {mu + sd:.3f}', alpha=0.7)

                    # 막대 색상 그라데이션
                    if len(n) > 0 and max(n) > 0:
                        cm = plt.cm.get_cmap('Blues')
                        for i, p in enumerate(patches):
                            p.set_facecolor(cm(0.4 + 0.6 * n[i] / max(n)))
                else:
                    ax.text(0.5, 0.5, 'No Class 26 data',
                            ha='center', va='center', transform=ax.transAxes,
                            fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

                ax.set_title(
                    f'Task {data["task_id"]}\nLearning: {data["learning_classes"]}\n'
                    f'Class 26: n={data["class_26_count"]}',
                    fontweight='bold', fontsize=11, pad=15
                )
                ax.set_xlabel('Score', fontsize=11, fontweight='bold')
                if idx == 0:
                    ax.set_ylabel('Density', fontsize=11, fontweight='bold')
                ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True,
                          fontsize=9, framealpha=0.9)
                ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
                ax.set_ylim(bottom=0)

            plt.tight_layout()
            save_path = os.path.join(class26_dir, f'{method.lower()}_class26_evolution.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"✓ {method} Class 26 evolution saved: {save_path}")
    
    def _create_class26_evolution_summary(self, tasks_sorted, all_methods, class26_dir):
        """Create Class 26 evolution summary analysis"""
        print("Creating Class 26 evolution summary analysis...")
        
        # Collect evolution data
        evolution_data = {}
        for method in all_methods:
            evolution_data[method] = {
                'tasks': [],
                'class26_means': [],
                'class26_stds': [],
                'ood_means': [],
                'ood_stds': [],
                'separability': [],
                'class26_counts': []
            }
        
        for task in tasks_sorted:
            task_id = task['task_id']
            for method in all_methods:
                if method in task['score_distributions']:
                    score_data = task['score_distributions'][method]
                    class_26_scores = np.array(score_data.get('class_26_scores', []))
                    ood_scores = np.array(score_data.get('ood_scores', []))
                    class_26_count = score_data.get('class_26_count', 0)
                    
                    if len(class_26_scores) > 0:
                        # Calculate metrics
                        class26_mean = class_26_scores.mean()
                        class26_std = class_26_scores.std()
                        
                        evolution_data[method]['tasks'].append(task_id)
                        evolution_data[method]['class26_means'].append(class26_mean)
                        evolution_data[method]['class26_stds'].append(class26_std)
                        evolution_data[method]['class26_counts'].append(class_26_count)
                        
                        if len(ood_scores) > 0:
                            ood_mean = ood_scores.mean()
                            ood_std = ood_scores.std()
                            separability = abs(class26_mean - ood_mean) / ((class26_std + ood_std) / 2 + 1e-8)
                            
                            evolution_data[method]['ood_means'].append(ood_mean)
                            evolution_data[method]['ood_stds'].append(ood_std)
                            evolution_data[method]['separability'].append(separability)
                        else:
                            evolution_data[method]['ood_means'].append(0)
                            evolution_data[method]['ood_stds'].append(0)
                            evolution_data[method]['separability'].append(0)
        
        # Create evolution plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Class 26 Score Evolution Analysis Summary', fontsize=18, fontweight='bold')
        
        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
        
        # 1. Class 26 mean score evolution
        for i, method in enumerate(sorted(all_methods)):
            data = evolution_data[method]
            if data['tasks']:
                ax1.plot(data['tasks'], data['class26_means'], marker='o', linewidth=3, 
                        markersize=8, color=colors[i % len(colors)], label=f'{method}',
                        markerfacecolor='white', markeredgewidth=2, alpha=0.8)
                
                # Add error bars
                if data['class26_stds']:
                    ax1.fill_between(data['tasks'], 
                                   np.array(data['class26_means']) - np.array(data['class26_stds']),
                                   np.array(data['class26_means']) + np.array(data['class26_stds']),
                                   color=colors[i % len(colors)], alpha=0.2)
        
        ax1.set_xlabel('Task Number', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Class 26 Mean Score', fontsize=12, fontweight='bold')
        ax1.set_title('Class 26 Score Evolution\n(Shows Catastrophic Forgetting)', fontweight='bold')
        ax1.legend(frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        ax1.set_facecolor('#FAFAFA')
        
        # 2. Separability evolution (only if OOD data exists)
        has_separability_data = any(data['separability'] and any(s > 0 for s in data['separability']) 
                                   for data in evolution_data.values())
        if has_separability_data:
            for i, method in enumerate(sorted(all_methods)):
                data = evolution_data[method]
                if data['tasks'] and data['separability'] and any(s > 0 for s in data['separability']):
                    ax2.plot(data['tasks'], data['separability'], marker='s', linewidth=3, 
                            markersize=8, color=colors[i % len(colors)], label=f'{method}',
                            markerfacecolor='white', markeredgewidth=2, alpha=0.8)
            
            ax2.set_xlabel('Task Number', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Class 26 vs OOD Separability', fontsize=12, fontweight='bold')
            ax2.set_title('Class 26 vs OOD Separability\n(Higher = Better)', fontweight='bold')
            ax2.legend(frameon=True, fancybox=True, shadow=True)
        else:
            ax2.text(0.5, 0.5, 'No OOD Separability Data Available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Class 26 vs OOD Separability\n(No Data)', fontweight='bold')
        
        ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        ax2.set_facecolor('#FAFAFA')
        
        # 3. Class 26 vs OOD mean comparison (only if OOD data exists)
        if has_separability_data:
            for i, method in enumerate(sorted(all_methods)):
                data = evolution_data[method]
                if data['tasks']:
                    ax3.plot(data['tasks'], data['class26_means'], marker='o', linewidth=3, 
                            markersize=8, color=colors[i % len(colors)], linestyle='-',
                            label=f'{method} Class 26', alpha=0.8,
                            markerfacecolor='white', markeredgewidth=2)
                    if data['ood_means'] and any(m > 0 for m in data['ood_means']):
                        ax3.plot(data['tasks'], data['ood_means'], marker='^', linewidth=3, 
                                markersize=8, color=colors[i % len(colors)], linestyle='--',
                                label=f'{method} OOD', alpha=0.6,
                                markerfacecolor='white', markeredgewidth=2)
            
            ax3.set_xlabel('Task Number', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Mean Score', fontsize=12, fontweight='bold')
            ax3.set_title('Class 26 vs OOD Mean Score Comparison', fontweight='bold')
            ax3.legend(frameon=True, fancybox=True, shadow=True)
        else:
            # Only show Class 26 evolution if no OOD data
            for i, method in enumerate(sorted(all_methods)):
                data = evolution_data[method]
                if data['tasks']:
                    ax3.plot(data['tasks'], data['class26_means'], marker='o', linewidth=3, 
                            markersize=8, color=colors[i % len(colors)], linestyle='-',
                            label=f'{method} Class 26', alpha=0.8,
                            markerfacecolor='white', markeredgewidth=2)
            
            ax3.set_xlabel('Task Number', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Class 26 Mean Score', fontsize=12, fontweight='bold')
            ax3.set_title('Class 26 Mean Score Evolution', fontweight='bold')
            ax3.legend(frameon=True, fancybox=True, shadow=True)
        
        ax3.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        ax3.set_facecolor('#FAFAFA')
        
        # 4. Class 26 sample count
        for i, method in enumerate(sorted(all_methods)):
            data = evolution_data[method]
            if data['tasks']:
                ax4.bar([t + i*0.1 for t in data['tasks']], data['class26_counts'], 
                       width=0.08, color=colors[i % len(colors)], alpha=0.8, 
                       label=f'{method}', edgecolor='white', linewidth=0.8)
        
        ax4.set_xlabel('Task Number', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Class 26 Sample Count', fontsize=12, fontweight='bold')
        ax4.set_title('Class 26 Sample Availability', fontweight='bold')
        ax4.legend(frameon=True, fancybox=True, shadow=True)
        ax4.grid(True, alpha=0.3, linestyle=':', linewidth=0.8, axis='y')
        ax4.set_facecolor('#FAFAFA')
        
        plt.tight_layout()
        save_path = os.path.join(class26_dir, 'class26_evolution_summary.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✓ Class 26 evolution summary saved: {save_path}")
    
    
        # ====================== t-SNE Feature Visualization ======================

    def create_tsne_for_task(
        self,
        task_id: int,
        id_features: np.ndarray,
        id_labels: np.ndarray,
        ood_features: np.ndarray = None,
        save_prefix: str = "tsne",
        random_state: int = 42,
        max_points: int = 6000,
        pca_dim: int = 50,
        tsne_perplexity: int = 30,
        tsne_iter: int = 1000,
    ):
        """
        Task별로 두 가지 이미지를 저장:
          1) ID 전용 t-SNE (클래스별 색) → {vis_dir}/task_{task_id}/tsne_id.png
          2) ID vs OOD t-SNE (이진 색)   → {vis_dir}/task_{task_id}/tsne_id_ood.png (OOD가 있을 때)
        """
        # Input validation
        if id_features is None or len(id_features) == 0:
            print(f"Warning: Empty ID features for Task {task_id}. Skipping T-SNE visualization.")
            return None, None
        
        if id_labels is not None and len(id_features) != len(id_labels):
            print(f"Warning: ID features length ({len(id_features)}) != ID labels length ({len(id_labels)}) for Task {task_id}")
            print(f"Adjusting to minimum length: {min(len(id_features), len(id_labels))}")
            min_len = min(len(id_features), len(id_labels))
            id_features = id_features[:min_len]
            id_labels = id_labels[:min_len] if id_labels is not None else None
        
        task_dir = os.path.join(self.vis_dir, f"task_{task_id}")
        os.makedirs(task_dir, exist_ok=True)

        # --- (1) ID-only t-SNE ---
        id_feats, id_labs = self._subsample(id_features, id_labels, max_points, random_state)
        
        # Safety check: ensure features and labels have same length
        if id_labs is not None and len(id_feats) != len(id_labs):
            print(f"Warning: Feature length ({len(id_feats)}) != Label length ({len(id_labs)}). Using min length.")
            min_len = min(len(id_feats), len(id_labs))
            id_feats = id_feats[:min_len]
            id_labs = id_labs[:min_len]
        
        emb_id = self._embed_tsne(id_feats, random_state, pca_dim, tsne_perplexity, tsne_iter)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_facecolor('#f8f9fa')
        classes = np.unique(id_labs) if id_labs is not None else np.array([0])
        palette = plt.cm.tab10(np.linspace(0, 1, max(len(classes), 1))) if len(classes) <= 10 \
                  else plt.cm.tab20(np.linspace(0, 1, len(classes)))

        for i, c in enumerate(classes):
            if id_labs is not None:
                m = (id_labs == c)
                # Safety check: ensure boolean mask has same length as embeddings
                if len(m) != len(emb_id):
                    print(f"Warning: Boolean mask length ({len(m)}) != Embedding length ({len(emb_id)})")
                    continue
            else:
                m = np.ones(len(emb_id), dtype=bool)
            
            ax.scatter(emb_id[m, 0], emb_id[m, 1],
                       s=18, c=[palette[i]], alpha=0.85, label=str(c),
                       edgecolors="white", linewidths=0.5)
        ax.set_title(f"Task {task_id} • ID t-SNE (classes={len(classes)})", fontweight="bold")
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.grid(True, alpha=0.25, linestyle="--")
        if len(classes) <= 12:
            ax.legend(title="Class", fontsize=8, frameon=True, fancybox=True, loc="best")
        id_only_path = os.path.join(task_dir, f"{save_prefix}_id.png")
        plt.tight_layout(); plt.savefig(id_only_path, dpi=300, bbox_inches="tight"); plt.close()

        # --- (2) ID vs OOD t-SNE ---
        id_ood_path = None
        if ood_features is not None and len(ood_features) > 0:
            id_feats2, _ = self._subsample(id_features, None, max_points // 2, random_state)
            ood_feats2, _ = self._subsample(ood_features, None, max_points // 2, random_state)

            # Safety check: ensure both arrays have valid shapes
            if len(id_feats2) == 0 or len(ood_feats2) == 0:
                print(f"Warning: Empty features after subsampling. ID: {len(id_feats2)}, OOD: {len(ood_feats2)}")
                return id_only_path, None

            all_feats = np.vstack([id_feats2, ood_feats2])
            emb_all = self._embed_tsne(all_feats, random_state, pca_dim, tsne_perplexity, tsne_iter)
            n_id = len(id_feats2)

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.set_facecolor('#f8f9fa')
            ax.scatter(emb_all[:n_id, 0], emb_all[:n_id, 1],
                       s=18, c="#1f77b4", alpha=0.75, label=f"ID ({n_id})",
                       edgecolors="white", linewidths=0.5, marker="o")
            ax.scatter(emb_all[n_id:, 0], emb_all[n_id:, 1],
                       s=22, c="#d62728", alpha=0.75, label=f"OOD ({len(emb_all)-n_id})",
                       edgecolors="white", linewidths=0.5, marker="^")
            ax.set_title(f"Task {task_id} • ID vs OOD t-SNE", fontweight="bold")
            ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.grid(True, alpha=0.25, linestyle="--")
            ax.legend(frameon=True, fancybox=True, loc="best")
            id_ood_path = os.path.join(task_dir, f"{save_prefix}_id_ood.png")
            plt.tight_layout(); plt.savefig(id_ood_path, dpi=300, bbox_inches="tight"); plt.close()

        print(f"✓ t-SNE saved: {id_only_path}" + (f", {id_ood_path}" if id_ood_path else ""))
        return id_only_path, id_ood_path

    # ------------------ helpers ------------------

    def _subsample(self, X: np.ndarray, y: np.ndarray = None, max_n: int = 6000, seed: int = 42):
        rng = np.random.RandomState(seed)
        if len(X) <= max_n:
            return X, y
        idx = rng.choice(len(X), size=max_n, replace=False)
        X_sub = X[idx]
        y_sub = y[idx] if y is not None else None
        return X_sub, y_sub

    def _embed_tsne(
        self,
        features: np.ndarray,
        random_state: int,
        pca_dim: int,
        tsne_perplexity: int,
        tsne_iter: int,
    ) -> np.ndarray:
        X = features
        if X.ndim != 2:
            X = X.reshape(len(X), -1)
        # 고차원일 때 PCA 선행
        if X.shape[1] > pca_dim:
            pca = PCA(n_components=pca_dim, random_state=random_state)
            X = pca.fit_transform(X)
        # perplexity는 샘플 수에 맞게 안전하게 조정
        perp = max(5, min(tsne_perplexity, (len(X)-1)//3 if len(X) >= 10 else 5))
        tsne = TSNE(n_components=2, perplexity=perp, n_iter=tsne_iter, random_state=random_state, verbose=0)
        return tsne.fit_transform(X)
    
    def _create_confusion_heatmaps(self):
        """
        experiment_results.json에 들어있는 task별/방법별 confusion_fpr95와 confusion_youdenJ를
        2x2 heatmap 이미지로 저장.
        """
        # self.results는 메모리에 이미 보유
        tasks = self.results.get('tasks', [])
        if not tasks:
            print("No tasks to visualize for confusion matrices.")
            return

        for task in tasks:
            task_id = task['task_id']
            ood_results = task.get('ood_results', {})

            for method, metrics in ood_results.items():
                if 'error' in metrics:
                    continue
                
                # FPR95 Confusion Matrix
                cf = metrics.get('confusion_fpr95')
                if cf:
                    self._create_single_confusion_matrix(task_id, method, cf, 'fpr95')
                
                # Youden's J Confusion Matrix
                cf_youden = metrics.get('confusion_youdenJ')
                if cf_youden:
                    self._create_single_confusion_matrix(task_id, method, cf_youden, 'youdenJ')
    
    def _create_single_confusion_matrix(self, task_id, method, cf, matrix_type):
        """Create a single confusion matrix visualization"""
        tp, fp, tn, fn = cf['tp'], cf['fp'], cf['tn'], cf['fn']
        cm = np.array([[tp, fn], [fp, tn]], dtype=np.int32)  # [[TP,FN],[FP,TN]]

        # 비율 표기를 위한 정규화(선택): 각 실제 클래스 행 기준
        with np.errstate(invalid='ignore', divide='ignore'):
            cm_row_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
            cm_row_norm = np.nan_to_num(cm_row_norm)

        # 그림 저장
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Enhanced confusion matrix with TP, FP, TN, FN labels
        sns.heatmap(
            cm_row_norm, annot=False, cmap='Blues', cbar=True, ax=ax,
            xticklabels=['Pred: ID', 'Pred: OOD'],
            yticklabels=['True: ID', 'True: OOD']
        )
        
        # Add TP, FP, TN, FN labels with values
        ax.text(0.5, 0.5, f'TP\n{tp}', ha='center', va='center', 
               fontsize=14, fontweight='bold', color='white' if tp > 0 else 'black',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='darkblue', alpha=0.7))
        
        ax.text(1.5, 0.5, f'FN\n{fn}', ha='center', va='center', 
               fontsize=14, fontweight='bold', color='white' if fn > 0 else 'black',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7))
        
        ax.text(0.5, 1.5, f'FP\n{fp}', ha='center', va='center', 
               fontsize=14, fontweight='bold', color='white' if fp > 0 else 'black',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.7))
        
        ax.text(1.5, 1.5, f'TN\n{tn}', ha='center', va='center', 
               fontsize=14, fontweight='bold', color='white' if tn > 0 else 'black',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='green', alpha=0.7))
        
        # Add performance metrics
        precision = cf['precision']
        recall = cf['recall']
        f1 = cf['f1']
        
        # Set title based on matrix type
        if matrix_type == 'fpr95':
            title = f"Task {task_id} • {method} • Confusion Matrix @ FPR95\n"
            title += f"Threshold: {cf['threshold']:.3f} | TPR: {cf['tpr']:.3f} | FPR: {cf['fpr']:.3f}\n"
            title += f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}"
            filename = f"task{task_id:02d}_{method}_confusion_fpr95.png"
        else:  # youdenJ
            youdenJ = cf['youdenJ']
            title = f"Task {task_id} • {method} • Confusion Matrix @ Youden's J\n"
            title += f"Threshold: {cf['threshold']:.3f} | TPR: {cf['tpr']:.3f} | FPR: {cf['fpr']:.3f}\n"
            title += f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | YoudenJ: {youdenJ:.3f}"
            filename = f"task{task_id:02d}_{method}_confusion_youdenJ.png"
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
        plt.tight_layout()

        out_path = os.path.join(self.vis_dir, filename)
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"✓ Confusion matrix saved: {filename}")
