 # %%
"""
GENERATE FIGURE 1: MODEL DEVELOPMENT PIPELINE
==============================================

This script creates a publication-quality figure showing the correct
model development pipeline with proper nesting of components within
cross-validation folds.

Outputs: figure_1_pipeline.png (300 DPI)
         figure_1_pipeline.pdf (vector format for publication)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import OUTPUT_DIR

# %%
# ==============================================================================
# OPTION 1: SIMPLE TEXT-BASED FLOWCHART (EASY TO MODIFY)
# ==============================================================================

def create_figure_1_simple():
    """
    Create a simple but clear flowchart of the pipeline.
    Best for: Quick creation, easy to understand, easy to modify
    """
    
    fig, ax = plt.subplots(figsize=(12, 16), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # Title
    ax.text(5, 19, 'Model Development Pipeline with 10-Fold Stratified Cross-Validation',
            fontsize=14, fontweight='bold', ha='center')
    
    # Dataset box
    rect_dataset = FancyBboxPatch((1.5, 17.5), 7, 1.2, 
                                  boxstyle="round,pad=0.1", 
                                  edgecolor='black', facecolor='#E8F4F8', linewidth=2)
    ax.add_patch(rect_dataset)
    ax.text(5, 18.5, 'DATASET: n=28,565 admission episodes', 
            fontsize=10, ha='center', fontweight='bold')
    ax.text(5, 18.05, '27 candidate predictors | 2,284 delirium cases (8.0%)',
            fontsize=9, ha='center', style='italic')
    
    # Arrow down
    ax.arrow(5, 17.5, 0, -0.4, head_width=0.2, head_length=0.15, fc='black', ec='black')
    
    # CV outer box (large rectangle)
    cv_rect = FancyBboxPatch((0.5, 0.5), 9, 16.5,
                             boxstyle="round,pad=0.15",
                             edgecolor='red', facecolor='#FFF8F0', 
                             linewidth=3, linestyle='--', alpha=0.3)
    ax.add_patch(cv_rect)
    ax.text(0.7, 16.8, '10-FOLD STRATIFIED CV', fontsize=11, fontweight='bold', color='red')
    
    # Left side: Training fold box
    train_rect = FancyBboxPatch((0.8, 8.5), 4, 7.5,
                               boxstyle="round,pad=0.08",
                               edgecolor='green', facecolor='#F0FFF0', linewidth=2)
    ax.add_patch(train_rect)
    ax.text(2.8, 15.7, 'TRAINING FOLD (90%)', fontsize=10, 
            fontweight='bold', ha='center', color='green')
    
    # Training fold steps
    training_steps = [
        '1. MICE Imputation',
        '   max_iterations=10, n_burn_in=3',
        '   random_state=42',
        '',
        '2. Standardization (numeric)',
        '   StandardScaler',
        '',
        '3. One-Hot Encoding (categorical)',
        '',
        '4. TWO-STAGE Feature Selection',
        '   Stage 1: Logistic Regression L1',
        '   Stage 2: Random Forest importance',
        '   → Stable core: 6 features (7 in final model)',
        '',
        '5. Hyperparameter Tuning',
        '   GridSearchCV (cv=5)',
        '',
        '6. Model Fitting',
        '   RF + Logistic Regression',
        '',
        '7. Calibration',
        '   Platt scaling (RF only)',
    ]
    
    y_pos = 15.2
    for step in training_steps:
        ax.text(1.2, y_pos, step, fontsize=8, ha='left', family='monospace')
        y_pos -= 0.35
    
    # Right side: Validation fold box
    val_rect = FancyBboxPatch((5.2, 8.5), 4, 7.5,
                             boxstyle="round,pad=0.08",
                             edgecolor='blue', facecolor='#F0F8FF', linewidth=2)
    ax.add_patch(val_rect)
    ax.text(7.2, 15.7, 'VALIDATION FOLD (10%)', fontsize=10,
            fontweight='bold', ha='center', color='blue')
    
    # Validation fold steps
    validation_steps = [
        '1. Apply MICE imputation',
        '   (fitted on training)',
        '',
        '2. Apply standardization',
        '   (fitted on training)',
        '',
        '3. Apply one-hot encoding',
        '   (fitted on training)',
        '',
        '4. Select same features',
        '',
        '5. Generate predictions',
        '',
        '6. Calculate metrics:',
        '   • ROC-AUC',
        '   • Accuracy',
        '   • Recall',
        '   • Precision',
        '   • F1-score',
        '',
    ]
    
    y_pos = 15.2
    for step in validation_steps:
        ax.text(5.6, y_pos, step, fontsize=8, ha='left', family='monospace')
        y_pos -= 0.35
    
    # Repeat label
    ax.text(5, 7.8, 'Repeat for all 10 folds, then average results',
            fontsize=9, ha='center', style='italic', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Final results box
    results_rect = FancyBboxPatch((1, 1), 8, 6.5,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='black', facecolor='#F5F5F5', linewidth=2)
    ax.add_patch(results_rect)
    ax.text(5, 7, 'FINAL RESULTS (Mean ± SD across 10 folds)',
            fontsize=10, ha='center', fontweight='bold')
    
    results_text = [
        'ROC-AUC: 0.893 (optimism-corrected)',
        'Accuracy: 0.XXX ± 0.XXX',
        'Recall: 0.XXX ± 0.XXX',
        'Precision: 0.XXX ± 0.XXX',
        'F1-score: 0.XXX ± 0.XXX',
        '',
        'Stable core features (n=6):',
        'Age, 4AT score, anticholinergic burden,',
        'anticholinergic drug count, behaviours of concern, dementia',
    ]
    
    y_pos = 6.5
    for text in results_text:
        ax.text(5, y_pos, text, fontsize=8, ha='center', family='monospace')
        y_pos -= 0.4
    
    # Add legend
    ax.text(0.5, -0.3, '● All preprocessing and model steps occur WITHIN each training fold',
            fontsize=8, style='italic')
    ax.text(0.5, -0.7, '● Preprocessors are fitted on training fold only, then applied to validation fold',
            fontsize=8, style='italic')
    ax.text(0.5, -1.1, '● This prevents information leakage and provides unbiased performance estimates',
            fontsize=8, style='italic')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'figure_1_pipeline_simple.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Figure saved: {output_path}")
    plt.show()

# %%
# ==============================================================================
# OPTION 2: DETAILED GRAPHVIZ DIAGRAM (PROFESSIONAL LOOK)
# ==============================================================================

def create_figure_1_graphviz():
    """
    Create a professional diagram using graphviz.
    Best for: Publication-quality, professional appearance
    Requires: pip install graphviz
    """
    
    try:
        from graphviz import Digraph
    except ImportError:
        print("Error: graphviz not installed. Install with: pip install graphviz")
        return
    
    # Create graph
    dot = Digraph(comment='Delirium Prediction Model Pipeline', format='png')
    dot.attr(rankdir='TB')
    dot.attr('graph', fontsize='10', fontname='Arial', dpi='300')
    
    # Styling
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='#E8F4F8', 
             fontname='Arial', fontsize='9', margin='0.2,0.1')
    
    # Dataset
    dot.node('dataset', 
             'DATASET\nn=28,565 episodes\n27 candidate predictors',
             fillcolor='#E8F4F8', fontsize='10', fontweight='bold')
    
    # CV Fold
    dot.node('cv_fold',
             '10-FOLD STRATIFIED CV\nStratifiedKFold(n_splits=10, shuffle=True)',
             fillcolor='#FFE8E8', fontsize='10', fontweight='bold')
    
    # Training steps
    dot.node('train_label', 'TRAINING FOLD (90%)', fillcolor='#E8FFE8', fontweight='bold')
    dot.node('mice', 
             'MICE Imputation\nmax_iter=10, n_burn_in=3\nrandom_state=42',
             fillcolor='#F0FFF0')
    dot.node('scale',
             'Standardization\nStandardScaler',
             fillcolor='#F0FFF0')
    dot.node('encode',
             'One-Hot Encoding\nOneHotEncoder',
             fillcolor='#F0FFF0')
    dot.node('fs_lasso',
             'Feature Selection - Stage 1\nLogisticRegressionCV (L1)\ncv=10',
             fillcolor='#F0FFF0')
    dot.node('fs_rf',
             'Feature Selection - Stage 2\nRandomForestClassifier\nn_estimators=100',
             fillcolor='#F0FFF0')
    dot.node('fs_final',
             'Final Features\nIntersection of both methods\n6-feature stable core (7 in final model)',
             fillcolor='#FFE8E8', fontweight='bold')
    dot.node('tune',
             'Hyperparameter Tuning\nGridSearchCV (cv=5)',
             fillcolor='#F0FFF0')
    dot.node('fit',
             'Model Fitting\nRandom Forest\nLogistic Regression',
             fillcolor='#F0FFF0')
    dot.node('calib',
             'Calibration\nPlatt scaling',
             fillcolor='#F0FFF0')
    
    # Validation steps
    dot.node('val_label', 'VALIDATION FOLD (10%)', fillcolor='#E8F0FF', fontweight='bold')
    dot.node('val_apply',
             'Apply Preprocessing\n(fitted on training)',
             fillcolor='#F0F8FF')
    dot.node('val_select',
             'Select Same Features\n(fitted on training)',
             fillcolor='#F0F8FF')
    dot.node('val_predict',
             'Generate Predictions',
             fillcolor='#F0F8FF')
    dot.node('val_metrics',
             'Calculate Metrics\nROC-AUC, Accuracy, Recall\nPrecision, F1-score',
             fillcolor='#F0F8FF')
    
    # Results
    dot.node('results',
             'FINAL RESULTS\nMean across 10 folds\nROC-AUC: 0.893 (optimism-corrected)\n6-feature stable core',
             fillcolor='#FFFACD', fontweight='bold')
    
    # Edges
    dot.edge('dataset', 'cv_fold')
    
    # Training path
    dot.edge('cv_fold', 'train_label')
    dot.edge('train_label', 'mice')
    dot.edge('mice', 'scale')
    dot.edge('scale', 'encode')
    dot.edge('encode', 'fs_lasso')
    dot.edge('fs_lasso', 'fs_rf')
    dot.edge('fs_rf', 'fs_final')
    dot.edge('fs_final', 'tune')
    dot.edge('tune', 'fit')
    dot.edge('fit', 'calib')
    
    # Validation path (parallel)
    dot.edge('cv_fold', 'val_label')
    dot.edge('val_label', 'val_apply')
    dot.edge('val_apply', 'val_select')
    dot.edge('val_select', 'val_predict')
    dot.edge('val_predict', 'val_metrics')
    
    # Convergence to results
    dot.edge('calib', 'results')
    dot.edge('val_metrics', 'results')
    
    # Save
    output_base = os.path.join(OUTPUT_DIR, 'figure_1_pipeline_graphviz')
    dot.render(output_base, format='png', cleanup=True)
    print(f"✓ Figure saved: {output_base}.png")

# %%
# ==============================================================================
# OPTION 3: MATPLOTLIB WITH DETAILED BOXES (MOST FLEXIBLE)
# ==============================================================================

def create_figure_1_detailed():
    """
    Create a detailed publication-quality figure with matplotlib.
    Best for: Most control, most customization
    """
    
    fig = plt.figure(figsize=(14, 18), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # Title
    title = ax.text(7, 19.3, 'Delirium Risk Prediction Model: Pipeline with 10-Fold Stratified Cross-Validation',
                   fontsize=13, fontweight='bold', ha='center')
    subtitle = ax.text(7, 18.8, 
                      'All preprocessing, feature selection, and model fitting occur within cross-validation folds to prevent information leakage',
                      fontsize=9, ha='center', style='italic')
    
    # Dataset
    dataset_box = FancyBboxPatch((2, 17.5), 10, 0.9,
                                boxstyle="round,pad=0.05",
                                edgecolor='black', facecolor='#E8F4F8', linewidth=2)
    ax.add_patch(dataset_box)
    ax.text(7, 18.15, 'Dataset: n=28,565 admission episodes | 27 candidate predictors | 2,284 delirium cases (8.0%)',
           fontsize=10, ha='center', fontweight='bold')
    
    # Arrow
    ax.arrow(7, 17.5, 0, -0.5, head_width=0.3, head_length=0.15, fc='black', ec='black', linewidth=2)
    
    # CV Container (large box with dashed border)
    cv_border = FancyBboxPatch((0.5, 1), 13, 16,
                              boxstyle="round,pad=0.1",
                              edgecolor='red', facecolor='none',
                              linewidth=3, linestyle='--')
    ax.add_patch(cv_border)
    ax.text(0.7, 16.7, '10-FOLD STRATIFIED CROSS-VALIDATION',
           fontsize=11, fontweight='bold', color='red', ha='left')
    
    # Training fold (left)
    train_box = FancyBboxPatch((1, 2), 5.5, 14.5,
                              boxstyle="round,pad=0.08",
                              edgecolor='green', facecolor='#F0FFF0',
                              linewidth=2)
    ax.add_patch(train_box)
    
    ax.text(3.75, 16.2, 'TRAINING FOLD (90%)',
           fontsize=10, fontweight='bold', ha='center', color='green',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Training steps
    train_content = """Fit on training data:

1. MICE Imputation
   • Algorithm: Iterative imputer
   • max_iterations = 10
   • n_burn_in = 3
   • random_state = 42
   • Numeric: age, 4AT, ACB, n_drugs
   • Categorical: mode

2. Standardization (StandardScaler)

3. One-Hot Encoding

4. TWO-STAGE Feature Selection
   a) LogisticRegressionCV (L1)
      cv=10, random_state=42
   b) RandomForest importance
      n_estimators=100
   c) Intersection → 6-feature stable core

5. Hyperparameter Tuning
   • GridSearchCV (cv=5)
   • Scoring: ROC-AUC

6. Model Fitting
   • Random Forest
   • Logistic Regression

7. Calibration (Platt scaling)"""
    
    ax.text(1.3, 15.5, train_content, fontsize=7.5, ha='left', va='top',
           family='monospace',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.3, pad=0.5))
    
    # Validation fold (right)
    val_box = FancyBboxPatch((7.5, 2), 5.5, 14.5,
                            boxstyle="round,pad=0.08",
                            edgecolor='blue', facecolor='#F0F8FF',
                            linewidth=2)
    ax.add_patch(val_box)
    
    ax.text(10.25, 16.2, 'VALIDATION FOLD (10%)',
           fontsize=10, fontweight='bold', ha='center', color='blue',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Validation steps
    val_content = """Apply fitted preprocessing
to validation data:

1. Apply MICE imputation
   (from training fold)

2. Apply standardization
   (from training fold)

3. Apply one-hot encoding
   (from training fold)

4. Select same features

5. Generate predictions

6. Calculate metrics:
   • ROC-AUC (primary)
   • Accuracy
   • Recall
   • Precision
   • F1-score
   • Calibration error

7. Record fold results"""
    
    ax.text(7.8, 15.5, val_content, fontsize=7.5, ha='left', va='top',
           family='monospace',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.3, pad=0.5))
    
    # Repeat label
    ax.text(7, 1.2, '↻ Repeat for all 10 folds, then calculate mean and standard deviation',
           fontsize=9, ha='center', fontweight='bold', style='italic',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.4, pad=0.5))
    
    # Results box (below)
    results_y = 0.2
    results_box = FancyBboxPatch((1.5, results_y-1.2), 11, 0.8,
                                boxstyle="round,pad=0.05",
                                edgecolor='black', facecolor='#FFFACD', linewidth=2)
    
    # Note about figure
    note = "Figure 1. Pipeline details: MICE (Multiple Imputation by Chained Equations) imputes missing numeric values. Two-stage feature selection combines " \
           "L1-regularized logistic regression and Random Forest importance. All preprocessing occurs within training folds only; validation folds use " \
           "fitted preprocessing from their paired training fold. Results are averaged across 10 folds to obtain final performance estimates."
    
    ax.text(7, -0.2, note, fontsize=7.5, ha='center', va='top', wrap=True,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5, pad=0.5))
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'figure_1_pipeline_detailed.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Figure saved: {output_path}")
    plt.show()

# %%
# ==============================================================================
# RUN TO GENERATE FIGURES
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print("GENERATING FIGURE 1: DELIRIUM PREDICTION MODEL PIPELINE")
    print("="*70)
    
    print("\n[1/3] Creating simple text-based flowchart...")
    create_figure_1_simple()
    
    print("\n[2/3] Creating detailed matplotlib diagram...")
    create_figure_1_detailed()
    
    print("\n[3/3] Attempting graphviz diagram...")
    try:
        create_figure_1_graphviz()
    except Exception as e:
        print(f"Note: Graphviz diagram skipped ({e})")
    
    print("\n" + "="*70)
    print("✓ FIGURES GENERATED SUCCESSFULLY")
    print("="*70)
    print("\nFiles created:")
    print("  • figure_1_pipeline_simple.png")
    print("  • figure_1_pipeline_detailed.png")
    print("  • figure_1_pipeline_graphviz.png (optional)")
    print("\nAll figures generated at 300 DPI for publication quality.")
    print("\nNext steps:")
    print("  1. Open the PNG files to review")
    print("  2. Choose the version you prefer")
    print("  3. Copy to your manuscript")
    print("  4. Update caption with specifics from your actual results")

sys.path.insert(0, os.path.dirname(__file__))
from common import OUTPUT_DIR


