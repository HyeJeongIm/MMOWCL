import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime


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
            'ood_results': task_info['ood_results']
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
                else:
                    row[f'{method}_AUROC'] = 0
                    row[f'{method}_FPR95'] = 100
            
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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Continual Learning Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. CL Accuracy Trend
        ax1.plot(df['Task'], df['CL_Accuracy'], marker='o', linewidth=3,
                markersize=10, color='#e74c3c', markerfacecolor='white',
                markeredgewidth=2, label='CL Accuracy')
        ax1.fill_between(df['Task'], df['CL_Accuracy'], alpha=0.3, color='#e74c3c')
        
        # Add value labels
        for i, (task, acc) in enumerate(zip(df['Task'], df['CL_Accuracy'])):
            ax1.annotate(f'{acc:.1f}%', (task, acc), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        ax1.set_xlabel('Task Number')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('CL Accuracy Trend Across Tasks', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 100)
        
        # 2. CL Performance Statistics
        stats_data = {
            'Average': df['CL_Accuracy'].mean(),
            'Best': df['CL_Accuracy'].max(),
            'Worst': df['CL_Accuracy'].min(),
            'Final': df['CL_Accuracy'].iloc[-1]
        }
        
        bars = ax2.bar(stats_data.keys(), stats_data.values(), 
                      color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
        
        # Add value labels on bars
        for bar, value in zip(bars, stats_data.values()):
            ax2.text(bar.get_x() + bar.get_width()/2., value + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('CL Performance Statistics', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        save_path = os.path.join(self.vis_dir, '02_continual_learning_performance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ CL performance saved: {save_path}")
    
    def _create_ood_performance_visualization(self, df):
        """Create OOD performance visualization (AUROC + FPR95 charts)"""
        ood_auroc_cols = [col for col in df.columns if col.endswith('_AUROC')]
        ood_fpr_cols = [col for col in df.columns if col.endswith('_FPR95')]
        
        if not ood_auroc_cols:
            return
        
        # Filter out tasks with no OOD samples
        df_with_ood = df[df['OOD_Test_Samples'] > 0].copy()
        if len(df_with_ood) == 0:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('OOD Detection Performance Visualization', fontsize=16, fontweight='bold')
        
        methods = [col.replace('_AUROC', '') for col in ood_auroc_cols]
        colors = ['#3498db', '#2ecc71', '#9b59b6', '#e67e22']
        
        # 1. AUROC Bar Chart
        x = np.arange(len(df_with_ood))
        width = 0.8 / len(methods)
        
        for i, (method, color) in enumerate(zip(methods, colors)):
            values = df_with_ood[f'{method}_AUROC'].values
            bars = ax1.bar(x + i * width, values, width, label=method, 
                          color=color, alpha=0.8)
            
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
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 100)
        
        # 2. AUROC Line Chart
        for i, col in enumerate(ood_auroc_cols):
            method_name = col.replace('_AUROC', '')
            ax2.plot(df_with_ood['Task'], df_with_ood[col], 
                    marker='o', linewidth=2.5, markersize=8,
                    color=colors[i % len(colors)], label=method_name)
        
        ax2.set_xlabel('Task Number')
        ax2.set_ylabel('AUROC (%)')
        ax2.set_title('AUROC Trends (Higher = Better)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(0, 100)
        
        # 3. FPR95 Bar Chart
        for i, (method, color) in enumerate(zip(methods, colors)):
            fpr_col = f'{method}_FPR95'
            if fpr_col in df_with_ood.columns:
                values = df_with_ood[fpr_col].values
                bars = ax3.bar(x + i * width, values, width, label=method, 
                              color=color, alpha=0.8)
                
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
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim(0, 100)
        
        # 4. FPR95 Line Chart
        for i, col in enumerate(ood_fpr_cols):
            method_name = col.replace('_FPR95', '')
            if col in df_with_ood.columns:
                ax4.plot(df_with_ood['Task'], df_with_ood[col], 
                        marker='s', linewidth=2.5, markersize=8,
                        color=colors[i % len(colors)], label=method_name)
        
        ax4.set_xlabel('Task Number')
        ax4.set_ylabel('FPR95 (%)')
        ax4.set_title('FPR95 Trends (Lower = Better)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_ylim(0, 100)
        
        plt.tight_layout()
        save_path = os.path.join(self.vis_dir, '03_ood_performance_visualization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ OOD performance visualization saved: {save_path}")
    
    def _create_ood_performance_tables(self, df):
        """Create OOD performance tables (AUROC + FPR95 numerical tables)"""
        ood_auroc_cols = [col for col in df.columns if col.endswith('_AUROC')]
        ood_fpr_cols = [col for col in df.columns if col.endswith('_FPR95')]
        
        if not ood_auroc_cols:
            return
        
        # Filter out tasks with no OOD samples
        df_with_ood = df[df['OOD_Test_Samples'] > 0].copy()
        if len(df_with_ood) == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('OOD Detection Performance Tables', fontsize=16, fontweight='bold')
        
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
        
        plt.tight_layout()
        save_path = os.path.join(self.vis_dir, '04_ood_performance_tables.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ OOD performance tables saved: {save_path}")