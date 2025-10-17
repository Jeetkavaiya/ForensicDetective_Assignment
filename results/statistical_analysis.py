"""
Statistical Analysis for Forensic Detective Project
Analyzes performance metrics across multiple classification models
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load performance metrics
data = {
    'model': ['XGBoost', 'XGBoost', 'XGBoost', 'XGBoost', 'XGBoost',
              'SVM', 'SVM', 'SVM', 'SVM', 'SVM',
              'SGD', 'SGD', 'SGD', 'SGD', 'SGD',
              'RandomForest', 'RandomForest', 'RandomForest', 'RandomForest', 'RandomForest'],
    'class': ['google', 'html', 'latex', 'python', 'word',
              'google_docs', 'html', 'latex', 'python', 'word',
              'google_docs_pdfs', 'html_pdfs', 'latex_pdfs', 'python_pdfs', 'word_pdfs',
              'google', 'html', 'latex', 'python', 'word'],
    'precision': [1.00, 0.92, 0.96, 0.95, 0.91,
                  1.00, 1.00, 0.85, 0.99, 0.89,
                  1.00, 1.00, 0.85, 0.87, 0.87,
                  1.00, 0.93, 0.93, 0.94, 0.92],
    'recall': [1.00, 0.98, 0.90, 0.89, 0.97,
               1.00, 1.00, 1.00, 0.70, 0.99,
               1.00, 1.00, 0.96, 0.68, 0.94,
               1.00, 0.95, 0.91, 0.90, 0.96],
    'f1_score': [1.00, 0.95, 0.93, 0.92, 0.94,
                 1.00, 1.00, 0.92, 0.82, 0.94,
                 1.00, 1.00, 0.90, 0.77, 0.90,
                 1.00, 0.94, 0.92, 0.92, 0.94],
    'support': [1412, 1250, 1250, 1250, 1250,
                1130, 1000, 1000, 1000, 1000,
                1412, 1250, 1250, 1250, 1250,
                1412, 1250, 1250, 1250, 1250],
    'accuracy': [0.95, 0.95, 0.95, 0.95, 0.95,
                 0.94, 0.94, 0.94, 0.94, 0.94,
                 0.92, 0.92, 0.92, 0.92, 0.92,
                 0.95, 0.95, 0.95, 0.95, 0.95]
}

df = pd.DataFrame(data)

print("=" * 80)
print("STATISTICAL ANALYSIS - FORENSIC DETECTIVE PROJECT")
print("=" * 80)
print()

# 1. Descriptive Statistics
print("1. DESCRIPTIVE STATISTICS BY MODEL")
print("-" * 80)
descriptive_stats = df.groupby('model')[['precision', 'recall', 'f1_score', 'accuracy']].agg([
    'mean', 'std', 'min', 'max', 'median'
])
print(descriptive_stats)
print()

# 2. Overall Model Performance Summary
print("2. OVERALL MODEL PERFORMANCE SUMMARY")
print("-" * 80)
model_summary = df.groupby('model')[['precision', 'recall', 'f1_score']].mean()
model_summary['avg_performance'] = model_summary.mean(axis=1)
model_summary = model_summary.sort_values('avg_performance', ascending=False)
print(model_summary)
print()

# 3. Class-wise Performance Analysis
print("3. CLASS-WISE PERFORMANCE ANALYSIS")
print("-" * 80)
# Normalize class names for comparison
df['normalized_class'] = df['class'].str.replace('_pdfs', '').str.replace('_docs', '')
class_performance = df.groupby('normalized_class')[['precision', 'recall', 'f1_score']].agg(['mean', 'std'])
print(class_performance)
print()

# 4. Statistical Tests - Friedman Test (non-parametric)
print("4. FRIEDMAN TEST (Non-parametric ANOVA)")
print("-" * 80)
print("Testing if there are significant differences between models")
print()

# Get F1-scores for each model (comparing across classes)
models = ['XGBoost', 'SVM', 'SGD', 'RandomForest']
f1_by_model = []
for model in models:
    f1_by_model.append(df[df['model'] == model]['f1_score'].values)

# Friedman test
stat, p_value = friedmanchisquare(*f1_by_model)
print(f"Friedman Chi-Square Statistic: {stat:.4f}")
print(f"P-value: {p_value:.4f}")
if p_value < 0.05:
    print("Result: SIGNIFICANT differences exist between models (p < 0.05)")
else:
    print("Result: NO significant differences between models (p >= 0.05)")
print()

# 5. Pairwise Comparisons (Wilcoxon Signed-Rank Test)
print("5. PAIRWISE MODEL COMPARISONS (Wilcoxon Signed-Rank Test)")
print("-" * 80)
from itertools import combinations

for model1, model2 in combinations(models, 2):
    f1_model1 = df[df['model'] == model1]['f1_score'].values
    f1_model2 = df[df['model'] == model2]['f1_score'].values
    
    # Handle different sizes
    min_len = min(len(f1_model1), len(f1_model2))
    stat, p = wilcoxon(f1_model1[:min_len], f1_model2[:min_len])
    
    print(f"{model1} vs {model2}:")
    print(f"  Statistic: {stat:.4f}, P-value: {p:.4f}", end="")
    if p < 0.05:
        print(" - SIGNIFICANT difference")
    else:
        print(" - NO significant difference")
print()

# 6. Confusion Matrix Statistics
print("6. CONFUSION MATRIX ANALYSIS")
print("-" * 80)

# Calculate error rates per class
df['error_rate'] = 1 - df['recall']
error_analysis = df.groupby('model')['error_rate'].agg(['mean', 'std', 'max'])
print("Error Rates by Model:")
print(error_analysis)
print()

# 7. Correlation Analysis
print("7. CORRELATION ANALYSIS")
print("-" * 80)
correlation_matrix = df[['precision', 'recall', 'f1_score']].corr()
print("Correlation between metrics:")
print(correlation_matrix)
print()

# 8. Best and Worst Performing Classes
print("8. BEST AND WORST PERFORMING CLASSES")
print("-" * 80)
print("Top 5 Best Performing (by F1-Score):")
best_classes = df.nlargest(5, 'f1_score')[['model', 'class', 'f1_score', 'precision', 'recall']]
print(best_classes.to_string(index=False))
print()

print("Top 5 Worst Performing (by F1-Score):")
worst_classes = df.nsmallest(5, 'f1_score')[['model', 'class', 'f1_score', 'precision', 'recall']]
print(worst_classes.to_string(index=False))
print()

# 9. Variance Analysis
print("9. VARIANCE ANALYSIS")
print("-" * 80)
variance_stats = df.groupby('model')[['precision', 'recall', 'f1_score']].var()
print("Variance by Model (lower = more consistent):")
print(variance_stats)
print()

# 10. Statistical Summary
print("10. KEY FINDINGS SUMMARY")
print("=" * 80)
best_model = model_summary.index[0]
best_score = model_summary.iloc[0]['avg_performance']
print(f"• Best Overall Model: {best_model} (Avg Performance: {best_score:.4f})")

most_consistent = variance_stats.mean(axis=1).idxmin()
print(f"• Most Consistent Model: {most_consistent}")

hardest_class = df.groupby('normalized_class')['f1_score'].mean().idxmin()
print(f"• Most Difficult Class to Classify: {hardest_class}")

easiest_class = df.groupby('normalized_class')['f1_score'].mean().idxmax()
print(f"• Easiest Class to Classify: {easiest_class}")

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

# Optional: Generate visualizations
def generate_visualizations():
    """Generate statistical visualizations"""
    
    # 1. Model Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Precision comparison
    df.boxplot(column='precision', by='model', ax=axes[0, 0])
    axes[0, 0].set_title('Precision Distribution by Model')
    axes[0, 0].set_xlabel('Model')
    axes[0, 0].set_ylabel('Precision')
    
    # Recall comparison
    df.boxplot(column='recall', by='model', ax=axes[0, 1])
    axes[0, 1].set_title('Recall Distribution by Model')
    axes[0, 1].set_xlabel('Model')
    axes[0, 1].set_ylabel('Recall')
    
    # F1-Score comparison
    df.boxplot(column='f1_score', by='model', ax=axes[1, 0])
    axes[1, 0].set_title('F1-Score Distribution by Model')
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_ylabel('F1-Score')
    
    # Overall comparison
    model_means = df.groupby('model')[['precision', 'recall', 'f1_score']].mean()
    model_means.plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_title('Average Metrics by Model')
    axes[1, 1].set_xlabel('Model')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend(loc='lower right')
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('statistical_analysis.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'statistical_analysis.png'")

# Uncomment to generate visualizations
generate_visualizations()