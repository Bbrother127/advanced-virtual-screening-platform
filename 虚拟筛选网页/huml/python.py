import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings

warnings.filterwarnings('ignore')

# 设置英文字体
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子确保可重复性
np.random.seed(42)


class ANXA1VirtualScreening:  # 修改：修正类名，不能以数字开头
    def __init__(self):
        self.models = {}
        self.results = {}
        self.detailed_results = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.training_times = {}
        self.smiles_column = None
        self.feature_names = []
        self.training_smiles = None
        self.food_smiles = None

    def is_flavonoid_from_smiles(self, smiles):
        """
        Identify flavonoid compounds from SMILES strings
        """
        if not isinstance(smiles, str):
            return False

        smiles_lower = smiles.lower()

        # Flavonoid characteristic patterns
        flavonoid_patterns = [
            'c1ccc2c(c1)c(=o)cc(o2)',  # Benzopyrone core
            'c1cc2c(cc1)oc(=o)cc2',  # Flavonoid core variant
            'c1cc(cc(c1)o)o',  # Polyhydroxy benzene ring
            'c1cc(cc(c1)oc)o',  # Hydroxy-methoxy
            'c1cc(cc(c1)oc)oc',  # Poly-methoxy
        ]

        # Check for flavonoid features
        score = 0

        # 1. Check for flavonoid core structure
        for pattern in flavonoid_patterns[:2]:
            if pattern in smiles_lower:
                score += 3
                break

        # 2. Check hydroxyl group count
        oh_count = smiles_lower.count('o') - smiles_lower.count('oc') * 0.5
        if oh_count >= 2:
            score += 1
        if oh_count >= 3:
            score += 1

        # 3. Check aromatic ring count
        ring_count = sum(1 for char in smiles if char.isdigit())
        if ring_count >= 2:
            score += 1
        if ring_count >= 3:
            score += 1

        # 4. Check for common flavonoid substituents
        if 'oc' in smiles_lower:  # Methoxy group
            score += 0.5
        if 'n' in smiles_lower:  # Sometimes contains nitrogen
            score += 0.5

        # Determine if flavonoid (threshold can be adjusted)
        return score >= 2.5

    def analyze_flavonoids_in_results(self, food_results, top_n=50):
        """Analyze flavonoid compounds in screening results"""
        print(f"\n{'=' * 80}")
        print(f"🔬 Analyzing flavonoid compounds in top {top_n} compounds")
        print(f"{'=' * 80}")

        if food_results is None or len(food_results) == 0:
            print("No screening results to analyze")
            return 0

        # Get top N compounds
        top_compounds = food_results.head(top_n)

        flavonoid_count = 0
        flavonoid_compounds = []

        print("\nFlavonoid identification results:")
        print("-" * 100)

        for idx, row in top_compounds.iterrows():
            smiles = row.get('SMILES', '')
            compound_name = row.get('Compound_Name', f'Compound{row["Rank"]}')

            is_flavonoid = self.is_flavonoid_from_smiles(str(smiles))

            if is_flavonoid:
                flavonoid_count += 1
                flavonoid_compounds.append({
                    'Rank': row['Rank'],
                    'Name': compound_name,
                    'SMILES': smiles[:50] + '...' if len(str(smiles)) > 50 else smiles,
                    'Probability': row.get('Probability_Active', 0),
                    'Flavonoid': 'Yes'
                })

                print(f"✓ {row['Rank']:3d}. {compound_name[:30]:30s} Prob: {row.get('Probability_Active', 0):.3f}  Flavonoid: Yes")
            else:
                print(f"  {row['Rank']:3d}. {compound_name[:30]:30s} Prob: {row.get('Probability_Active', 0):.3f}  Flavonoid: No")

        print("-" * 100)
        print(f"\n📊 Statistics:")
        print(f"  Compounds analyzed: {top_n}")
        print(f"  Flavonoid compounds: {flavonoid_count} ({flavonoid_count / top_n * 100:.1f}%)")
        print(f"  Non-flavonoid compounds: {top_n - flavonoid_count} ({(top_n - flavonoid_count) / top_n * 100:.1f}%)")

        # Save flavonoid compounds
        if flavonoid_compounds:
            flavonoid_df = pd.DataFrame(flavonoid_compounds)
            filename = f'flavonoid_compounds_top_{top_n}.csv'
            flavonoid_df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"\n✅ Flavonoid compounds saved as '{filename}'")

        # Visualize flavonoid distribution
        self.visualize_flavonoid_distribution(flavonoid_count, top_n - flavonoid_count, top_n)

        return flavonoid_count

    def visualize_flavonoid_distribution(self, flavonoid_count, non_flavonoid_count, total_count):
        """Visualize flavonoid compound distribution"""
        # 修改：增加图形大小和调整布局
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 增加子图间距
        plt.subplots_adjust(wspace=0.3, left=0.08, right=0.95, top=0.9, bottom=0.15)

        # Pie chart
        labels = ['Flavonoid', 'Non-Flavonoid']
        sizes = [flavonoid_count, non_flavonoid_count]
        colors = ['#ff9999', '#66b3ff']

        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Flavonoid Distribution in Top {total_count} Compounds', fontsize=12, pad=20)

        # Bar chart
        x = np.arange(len(labels))
        bars = ax2.bar(x, sizes, color=colors, alpha=0.8)
        ax2.set_xlabel('Compound Type', fontsize=11, labelpad=10)
        ax2.set_ylabel('Count', fontsize=11, labelpad=10)
        ax2.set_title('Flavonoid vs Non-Flavonoid Compounds', fontsize=12, pad=20)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=10)

        # Add value labels on bars
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                     f'{size}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig('flavonoid_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✅ Flavonoid distribution plot saved as 'flavonoid_distribution.png'")

    def load_training_data(self):
        """Load training data"""
        print("Loading training data mydata.csv...")

        try:
            data = pd.read_csv('mydata.csv')
            print(f"Successfully loaded training {data.shape[0]} rows, {data.shape[1]} columns")

            column_names = data.columns.tolist()

            # Find SMILES column
            smiles_candidates = ['SMILES', 'smiles', 'Smiles']
            self.smiles_column = None
            for col in smiles_candidates:
                if col in column_names:
                    self.smiles_column = col
                    break

            if not self.smiles_column:
                self.smiles_column = column_names[0]

            # Find label column
            label_candidates = ['Labels', 'labels', 'Label', 'label']
            label_column = None
            for col in label_candidates:
                if col in column_names:
                    label_column = col
                    break

            if not label_column:
                label_column = column_names[-1]

            # Extract features
            smiles_data = data[self.smiles_column].values
            X = self.generate_descriptors_from_smiles(smiles_data)

            # Extract labels
            y = data[label_column].values

            # Ensure labels are 0/1 format
            unique_labels = np.unique(y)
            if set(unique_labels) != {0, 1}:
                if all(isinstance(val, str) for val in unique_labels):
                    label_map = {}
                    for val in unique_labels:
                        val_lower = str(val).lower()
                        if val_lower in ['active', '1', '+', 'yes', 'y', 'true']:
                            label_map[val] = 1
                        else:
                            label_map[val] = 0
                    y = np.array([label_map[val] for val in y])
                elif set(unique_labels) == {1, -1}:
                    y = np.array([1 if val == 1 else 0 for val in y])

            # Check for missing values
            if np.isnan(X).any():
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                X = imputer.fit_transform(X)

            print(f"Final data shape: X={X.shape}, y={y.shape}")

            self.training_smiles = smiles_data
            return X, y

        except Exception as e:
            print(f"Error loading {str(e)}")
            raise

    def generate_descriptors_from_smiles(self, smiles_list):
        """Generate molecular descriptors from SMILES"""
        print("Generating molecular descriptors from SMILES...")

        try:
            # Try using rdkit
            try:
                from rdkit import Chem
                from rdkit.Chem import Descriptors
                from rdkit.Chem import rdMolDescriptors

                descriptors_list = []
                for smiles in smiles_list:
                    if isinstance(smiles, str):
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None:
                            desc = [
                                Descriptors.MolWt(mol),
                                Descriptors.MolLogP(mol),
                                Descriptors.NumHAcceptors(mol),
                                Descriptors.NumHDonors(mol),
                                Descriptors.NumRotatableBonds(mol),
                                Descriptors.TPSA(mol),
                                Descriptors.NumAromaticRings(mol),
                                Descriptors.NumAliphaticRings(mol),
                                rdMolDescriptors.CalcNumRings(mol),
                                Descriptors.FractionCSP3(mol),
                            ]
                            descriptors_list.append(desc)
                        else:
                            descriptors_list.append([0] * 10)
                    else:
                        descriptors_list.append([0] * 10)

                X = np.array(descriptors_list)
                self.feature_names = ['MW', 'LogP', 'HBA', 'HBD', 'RotBonds',
                                      'TPSA', 'AroRings', 'AliRings', 'TotalRings', 'Fsp3']
                print("Successfully generated descriptors using rdkit")

            except ImportError:
                # rdkit not available, use simplified version
                print("rdkit not installed, using simplified descriptors")
                descriptors_list = []
                for smiles in smiles_list:
                    if isinstance(smiles, str):
                        desc = [
                            len(str(smiles)),
                            str(smiles).count('C'),
                            str(smiles).count('O'),
                            str(smiles).count('N'),
                            str(smiles).count('('),
                            str(smiles).count('='),
                            str(smiles).count('#'),
                            sum(1 for c in str(smiles) if c.isdigit()),
                            str(smiles).count('o'),
                            str(smiles).count('c'),
                        ]
                        descriptors_list.append(desc)
                    else:
                        descriptors_list.append([0] * 10)

                X = np.array(descriptors_list)
                self.feature_names = ['SMILES_Length', 'C_atoms', 'O_atoms', 'N_atoms', 'Branches',
                                      'Double_bonds', 'Triple_bonds', 'Rings', 'OH_groups', 'Aromatic_C']

            print(f"Generated {X.shape[1]} descriptor features")
            return X

        except Exception as e:
            print(f"Error generating descriptors: {str(e)}")
            X = np.random.randn(len(smiles_list), 10)
            self.feature_names = [f'Feature_{i + 1}' for i in range(10)]
            return X

    def initialize_models(self):
        """Initialize 6 machine learning models"""
        self.models = {
            'SVM': {
                'model': SVC(probability=True, random_state=42, class_weight='balanced'),
                'description': 'Support Vector Machine',
                'color': '#1f77b4'
            },
            'LR': {
                'model': LogisticRegression(random_state=42, max_iter=1000,
                                            class_weight='balanced'),
                'description': 'Logistic Regression',
                'color': '#ff7f0e'
            },
            'KNN': {
                'model': KNeighborsClassifier(n_neighbors=5, weights='distance'),
                'description': 'K-Nearest Neighbors',
                'color': '#2ca02c'
            },
            'RF': {
                'model': RandomForestClassifier(random_state=42, n_estimators=100,
                                               class_weight='balanced'),
                'description': 'Random Forest',
                'color': '#d62728'
            },
            'PLS-DA': {
                'model': LinearDiscriminantAnalysis(),
                'description': 'Partial Least Squares Discriminant Analysis',
                'color': '#9467bd'
            },
            'DFNN': {
                'model': MLPClassifier(hidden_layer_sizes=(64, 32),
                                      random_state=42,
                                      max_iter=1000,
                                      early_stopping=True,
                                      validation_fraction=0.1),
                'description': 'Deep Feedforward Neural Network',
                'color': '#8c564b'
            }
        }

        print(f"\n{'=' * 80}")
        print("Initializing 6 machine learning models")
        print(f"{'=' * 80}")

    def train_and_compare_models(self, X_train, X_test, y_train, y_test):
        """Train and compare all models"""
        print(f"\n{'=' * 80}")
        print("Training and evaluating all models...")
        print(f"{'=' * 80}")

        for name, info in self.models.items():
            print(f"\n{'=' * 60}")
            print(f"🔄 Training {name} model...")

            try:
                import time
                start_time = time.time()
                info['model'].fit(X_train, y_train)
                train_time = time.time() - start_time

                start_time = time.time()
                y_pred = info['model'].predict(X_test)
                predict_time = time.time() - start_time

                if hasattr(info['model'], 'predict_proba'):
                    y_pred_proba = info['model'].predict_proba(X_test)[:, 1]
                else:
                    y_pred_proba = None

                metrics = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, zero_division=0),
                    'Recall': recall_score(y_test, y_pred, zero_division=0),
                    'F1_Score': f1_score(y_test, y_pred, zero_division=0),
                    'Training_Time(s)': train_time,
                    'Prediction_Time(s)': predict_time
                }

                if y_pred_proba is not None and len(np.unique(y_test)) > 1:
                    try:
                        metrics['AUC_ROC'] = roc_auc_score(y_test, y_pred_proba)
                        metrics['AUPRC'] = average_precision_score(y_test, y_pred_proba)

                        # Calculate ROC curve
                        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                        metrics['roc_curve'] = (fpr, tpr)
                    except:
                        metrics['AUC_ROC'] = 0.5
                        metrics['AUPRC'] = 0.0

                cv_scores = cross_val_score(info['model'], X_train, y_train,
                                            cv=5, scoring='f1', n_jobs=-1)
                metrics['CV_F1_Mean'] = cv_scores.mean()
                metrics['CV_F1_Std'] = cv_scores.std()

                cm = confusion_matrix(y_test, y_pred)
                metrics['confusion_matrix'] = cm

                if cm.shape == (2, 2):
                    metrics['TN'], metrics['FP'], metrics['FN'], metrics['TP'] = cm.ravel()
                    metrics['Specificity'] = metrics['TN'] / (metrics['TN'] + metrics['FP']) if (metrics['TN'] + metrics['FP']) > 0 else 0

                # Feature importance
                if hasattr(info['model'], 'feature_importances_'):
                    self.feature_importance[name] = info['model'].feature_importances_

                self.results[name] = metrics
                self.detailed_results[name] = {
                    'metrics': metrics,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'confusion_matrix': cm,
                    'model': info['model']
                }

                print(f"✅ {name} training completed!")
                print(f"   Training time: {metrics['Training_Time(s)']:.3f}s")
                print(f"   F1 Score: {metrics['F1_Score']:.4f}")
                print(f"   AUC-ROC: {metrics.get('AUC_ROC', 0.5):.4f}")

            except Exception as e:
                print(f"❌ {name} training failed: {str(e)}")
                self.results[name] = {
                    'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1_Score': 0,
                    'AUC_ROC': 0.5, 'Training_Time(s)': 0, 'Prediction_Time(s)': 0
                }

    def display_model_comparison_table(self):
        """Display model comparison table"""
        print(f"\n{'=' * 100}")
        print("📊 Model Performance Comparison")
        print(f"{'=' * 100}")

        comparison_data = []

        for name, metrics in self.results.items():
            row = {
                'Model': name,
                'F1_Score': f"{metrics.get('F1_Score', 0):.4f}",
                'AUC_ROC': f"{metrics.get('AUC_ROC', 0.5):.4f}",
                'Accuracy': f"{metrics.get('Accuracy', 0):.4f}",
                'Precision': f"{metrics.get('Precision', 0):.4f}",
                'Recall': f"{metrics.get('Recall', 0):.4f}",
                'Specificity': f"{metrics.get('Specificity', 0):.4f}",
                'CV_F1_Mean': f"{metrics.get('CV_F1_Mean', 0):.4f}",
                'Training_Time(s)': f"{metrics.get('Training_Time(s)', 0):.3f}"
            }
            comparison_data.append(row)

        df_comparison = pd.DataFrame(comparison_data)
        df_comparison['F1_Score_num'] = df_comparison['F1_Score'].astype(float)
        df_comparison = df_comparison.sort_values('F1_Score_num', ascending=False)
        df_comparison = df_comparison.drop('F1_Score_num', axis=1)

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)

        print("\nModel Performance Ranking (by F1 Score):")
        print("-" * 100)
        print(df_comparison.to_string(index=False))
        print("-" * 100)

        df_comparison.to_csv('model_comparison_table.csv', index=False, encoding='utf-8-sig')
        print("✅ Comparison table saved as 'model_comparison_table.csv'")

        return df_comparison

    def select_best_model(self):
        """Select the best model"""
        print(f"\n{'=' * 80}")
        print("🎯 Selecting Best Model")
        print(f"{'=' * 80}")

        best_model = None
        best_score = -1

        for name, metrics in self.results.items():
            auc_score = metrics.get('AUC_ROC', 0.5)
            f1_score_val = metrics.get('F1_Score', 0)

            composite_score = 0.5 * auc_score + 0.5 * f1_score_val

            if composite_score > best_score:
                best_score = composite_score
                self.best_model_name = name
                self.best_model = self.models[name]['model']

        print(f"\n🏆 Best Model: {self.best_model_name}")
        print(f"   Composite Score: {best_score:.4f}")

        best_metrics = self.results[self.best_model_name]
        print(f"\n📈 Best Model Detailed Metrics:")
        print(f"   F1 Score:     {best_metrics.get('F1_Score', 0):.4f}")
        print(f"   AUC-ROC:      {best_metrics.get('AUC_ROC', 0.5):.4f}")
        print(f"   Accuracy:     {best_metrics.get('Accuracy', 0):.4f}")
        print(f"   Precision:    {best_metrics.get('Precision', 0):.4f}")
        print(f"   Recall:       {best_metrics.get('Recall', 0):.4f}")
        print(f"   Specificity:  {best_metrics.get('Specificity', 0):.4f}")
        print(f"   Training Time: {best_metrics.get('Training_Time(s)', 0):.3f}s")

        return self.best_model_name, self.best_model

    def create_model_specific_chart(self, model_name, model_info):
        """Create performance chart for a single model"""
        if model_name not in self.results:
            return

        metrics = self.results[model_name]

        # 修改：增加图形大小和调整布局
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 修改：增加子图间距，防止标题重叠
        plt.subplots_adjust(
            left=0.08,  # 左边距
            right=0.95,  # 右边距
            bottom=0.08,  # 底边距
            top=0.92,  # 顶部边距
            wspace=0.3,  # 水平间距
            hspace=0.4  # 垂直间距
        )

        # 1. Main performance metrics
        ax1 = axes[0, 0]
        performance_metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'Specificity']
        performance_values = [metrics.get(m, 0) for m in performance_metrics]

        x_pos = np.arange(len(performance_metrics))
        bars = ax1.bar(x_pos, performance_values, color=model_info['color'], alpha=0.7)

        # 修改：调整坐标轴标签和标题间距
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(performance_metrics, rotation=30, ha='right', fontsize=10)
        ax1.set_ylabel('Score', fontsize=11, labelpad=10)
        ax1.set_title('Main Performance Metrics', fontsize=12, fontweight='bold', pad=15)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.tick_params(axis='y', labelsize=9)

        # Add value labels
        for bar, value in zip(bars, performance_values):
            height = bar.get_height()
            y_pos = height + 0.02 if height < 0.9 else height - 0.04
            ax1.text(bar.get_x() + bar.get_width() / 2., y_pos,
                     f'{value:.3f}', ha='center', va='bottom', fontsize=9)

        # 2. ROC curve
        ax2 = axes[0, 1]
        if 'roc_curve' in metrics:
            fpr, tpr = metrics['roc_curve']
            auc_score = metrics.get('AUC_ROC', 0.5)

            ax2.plot(fpr, tpr, color=model_info['color'], lw=2,
                     label=f'AUC = {auc_score:.3f}')
            ax2.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
            ax2.fill_between(fpr, tpr, alpha=0.2, color=model_info['color'])

            # 修改：调整坐标轴标签间距
            ax2.set_xlabel('False Positive Rate', fontsize=11, labelpad=10)
            ax2.set_ylabel('True Positive Rate', fontsize=11, labelpad=10)
            ax2.set_title('ROC Curve', fontsize=12, fontweight='bold', pad=15)
            ax2.legend(loc="lower right", fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='both', which='major', labelsize=9)
        else:
            ax2.text(0.5, 0.5, 'ROC curve not available',
                     ha='center', va='center', transform=ax2.transAxes, fontsize=11)
            ax2.set_title('ROC Curve', fontsize=12, fontweight='bold', pad=15)

        # 3. Confusion matrix
        ax3 = axes[1, 0]
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                        cbar_kws={'label': 'Count'})
            # 修改：调整混淆矩阵标题和标签间距
            ax3.set_title('Confusion Matrix', fontsize=12, fontweight='bold', pad=15)
            ax3.set_xlabel('Predicted Label', fontsize=11, labelpad=10)
            ax3.set_ylabel('True Label', fontsize=11, labelpad=10)
            ax3.tick_params(axis='both', labelsize=9)
        else:
            ax3.text(0.5, 0.5, 'Confusion matrix not available',
                     ha='center', va='center', transform=ax3.transAxes, fontsize=11)
            ax3.set_title('Confusion Matrix', fontsize=12, fontweight='bold', pad=15)

        # 4. Training time and cross-validation
        ax4 = axes[1, 1]
        train_time = metrics.get('Training_Time(s)', 0)
        cv_mean = metrics.get('CV_F1_Mean', 0)
        cv_std = metrics.get('CV_F1_Std', 0)

        # Create dual axes
        ax4_secondary = ax4.twinx()

        # Training time bar chart
        time_bar = ax4.bar(['Training\nTime'], [train_time],
                           color=model_info['color'], alpha=0.6, width=0.5)
        # 修改：调整坐标轴标签间距
        ax4.set_ylabel('Training Time (s)', color=model_info['color'],
                       fontsize=11, labelpad=10)
        ax4.tick_params(axis='y', labelcolor=model_info['color'])
        ax4.tick_params(axis='y', labelsize=9)

        # Cross-validation score scatter plot
        cv_scatter = ax4_secondary.scatter(['Cross-\nValidation'], [cv_mean],
                                           color='red', s=100, marker='o', zorder=5)
        ax4_secondary.errorbar(['Cross-\nValidation'], [cv_mean], yerr=[cv_std],
                               color='red', capsize=8, capthick=2, zorder=4)
        ax4_secondary.set_ylabel('F1 Score', color='red', fontsize=11, labelpad=10)
        ax4_secondary.tick_params(axis='y', labelcolor='red')
        ax4_secondary.set_ylim(0, 1.05)
        ax4_secondary.tick_params(axis='y', labelsize=9)

        ax4.set_title('Training Time and\nCross-Validation', fontsize=12,
                      fontweight='bold', pad=15)
        ax4.set_xticks([0])
        ax4.set_xticklabels([''], rotation=0)

        # Add value labels with adjusted positions
        ax4.text(0, train_time + max(train_time, 0.1) * 0.05,
                 f'{train_time:.2f}s', ha='center', va='bottom',
                 fontsize=9, color=model_info['color'])
        ax4_secondary.text(0, cv_mean + 0.05,
                           f'{cv_mean:.3f}±{cv_std:.3f}', ha='center', va='bottom',
                           fontsize=9, color='red')

        # 修改：调整图形主标题位置
        fig.suptitle(f'{model_name} - Performance Analysis',
                     fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为标题留出空间

        filename = f'{model_name}_performance.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        print(f"✅ {model_name} performance chart saved as '{filename}'")

    def create_individual_model_charts(self):
        """Create individual performance charts for each model"""
        print("\nGenerating individual model performance charts...")

        for name, info in self.models.items():
            if name in self.results:
                self.create_model_specific_chart(name, info)

    def create_radar_chart(self):
        """Create performance metrics radar chart"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='polar')

        # 修改：增加图形顶部空间
        plt.subplots_adjust(top=0.85)

        metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'Specificity']
        n_metrics = len(metrics_list)

        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]

        for name, info in self.models.items():
            if name in self.results:
                metrics = self.results[name]
                values = [metrics.get(m, 0) for m in metrics_list]
                values += values[:1]

                ax.plot(angles, values, 'o-', linewidth=2,
                        label=name, color=info['color'], alpha=0.7)
                ax.fill(angles, values, alpha=0.1, color=info['color'])

        # 修改：调整雷达图标签位置
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_list, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.set_title('Model Performance Radar Chart', fontsize=14,
                     fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig('model_performance_radar.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Model performance radar chart saved as 'model_performance_radar.png'")

    def create_roc_comparison_chart(self):
        """Create ROC curve comparison chart"""
        fig, ax = plt.subplots(figsize=(10, 8))

        # 修改：增加边距
        plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.9)

        for name, info in self.models.items():
            if name in self.results and 'roc_curve' in self.results[name]:
                fpr, tpr = self.results[name]['roc_curve']
                auc_score = self.results[name].get('AUC_ROC', 0.5)

                ax.plot(fpr, tpr, lw=2,
                        label=f'{name} (AUC = {auc_score:.3f})',
                        color=info['color'])

        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        # 修改：调整坐标轴标签间距
        ax.set_xlabel('False Positive Rate', fontsize=12, labelpad=10)
        ax.set_ylabel('True Positive Rate', fontsize=12, labelpad=10)
        ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=10)

        plt.tight_layout()
        plt.savefig('model_roc_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ ROC curves comparison chart saved as 'model_roc_comparison.png'")

    def create_performance_bar_chart(self):
        """Create performance comparison bar chart"""
        metrics_to_plot = ['F1_Score', 'AUC_ROC', 'Accuracy', 'Precision', 'Recall']

        # 修改：增加图形大小和间距
        fig, axes = plt.subplots(2, 3, figsize=(16, 12))
        axes = axes.flatten()

        # 修改：调整整体布局
        plt.subplots_adjust(
            left=0.08,
            right=0.95,
            bottom=0.08,
            top=0.92,
            wspace=0.3,  # 增加水平间距
            hspace=0.4  # 增加垂直间距
        )

        model_names = list(self.models.keys())
        colors = [self.models[name]['color'] for name in model_names]

        for idx, metric in enumerate(metrics_to_plot):
            if idx < len(axes):
                ax = axes[idx]

                values = []
                valid_models = []
                valid_colors = []

                for name in model_names:
                    if name in self.results:
                        value = self.results[name].get(metric, 0)
                        values.append(value)
                        valid_models.append(name)
                        valid_colors.append(self.models[name]['color'])

                if values:
                    bars = ax.bar(valid_models, values, color=valid_colors, alpha=0.8)
                    # 修改：调整坐标轴标签和标题间距
                    ax.set_ylabel('Score', fontsize=11, labelpad=10)
                    ax.set_title(f'{metric} Comparison', fontsize=12,
                                 fontweight='bold', pad=15)
                    ax.set_ylim(0, 1)
                    ax.tick_params(axis='x', rotation=45, labelsize=9)
                    ax.set_xlabel('Model', fontsize=11, labelpad=8)
                    ax.grid(True, alpha=0.3, axis='y')
                    ax.tick_params(axis='y', labelsize=9)

                    # Mark best model
                    if self.best_model_name in valid_models:
                        best_idx = valid_models.index(self.best_model_name)
                        bars[best_idx].set_edgecolor('black')
                        bars[best_idx].set_linewidth(2)

        # Last subplot: Model ranking
        ax_last = axes[-1]
        sorted_models = sorted(self.results.items(),
                               key=lambda x: x[1].get('F1_Score', 0),
                               reverse=True)

        rank_names = [m[0] for m in sorted_models]
        rank_scores = [m[1].get('F1_Score', 0) for m in sorted_models]
        rank_colors = [self.models[name]['color'] for name in rank_names]

        bars_last = ax_last.barh(rank_names, rank_scores, color=rank_colors)
        # 修改：调整标签间距
        ax_last.set_xlabel('F1 Score', fontsize=11, labelpad=10)
        ax_last.set_title('Model F1 Score Ranking', fontsize=12,
                          fontweight='bold', pad=15)
        ax_last.set_xlim(0, 1)
        ax_last.tick_params(axis='both', labelsize=9)

        # Mark best model
        if self.best_model_name in rank_names:
            best_idx = rank_names.index(self.best_model_name)
            bars_last[best_idx].set_edgecolor('black')
            bars_last[best_idx].set_linewidth(2)

        # 修改：调整主标题位置
        plt.suptitle('Model Performance Metrics Comparison',
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Model performance comparison chart saved as 'model_performance_comparison.png'")

    def create_training_time_chart(self):
        """Create training time comparison chart"""
        model_names = []
        train_times = []
        colors = []

        for name, info in self.models.items():
            if name in self.results:
                model_names.append(name)
                train_times.append(self.results[name].get('Training_Time(s)', 0))
                colors.append(info['color'])

        if train_times:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # 修改：调整间距
            plt.subplots_adjust(left=0.08, right=0.95, bottom=0.15, top=0.9, wspace=0.25)

            # Bar chart
            bars = ax1.bar(model_names, train_times, color=colors, alpha=0.8)
            ax1.set_xlabel('Model', fontsize=11, labelpad=10)
            ax1.set_ylabel('Training Time (s)', fontsize=11, labelpad=10)
            ax1.set_title('Model Training Time Comparison', fontsize=12,
                          fontweight='bold', pad=20)
            ax1.tick_params(axis='x', rotation=45, labelsize=9)
            ax1.tick_params(axis='y', labelsize=9)
            ax1.grid(True, alpha=0.3)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                         f'{height:.2f}s', ha='center', va='bottom', fontsize=9)

            # Pie chart
            ax2.pie(train_times, labels=model_names, colors=colors, autopct='%1.1f%%',
                    startangle=90, textprops={'fontsize': 9})
            ax2.set_title('Training Time Distribution', fontsize=12,
                          fontweight='bold', pad=20)

            plt.tight_layout()
            plt.savefig('model_training_time_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("✅ Training time comparison chart saved as 'model_training_time_comparison.png'")

    def create_comparison_visualizations(self):
        """Create model comparison visualizations"""
        print("\nGenerating model comparison visualizations...")

        # 1. Radar chart
        self.create_radar_chart()

        # 2. ROC curve comparison
        self.create_roc_comparison_chart()

        # 3. Performance bar chart
        self.create_performance_bar_chart()

        # 4. Training time comparison
        self.create_training_time_chart()

    def screen_food_compounds(self):
        """Screen food compounds"""
        print(f"\n{'=' * 60}")
        print("Loading screening library database.csv...")  # 修改：文件名从food.csv改为database.csv
        print(f"{'=' * 60}")

        try:
            food_data = pd.read_csv('database.csv')  # 修改：文件名从food.csv改为database.csv
            print(f"Successfully loaded screening library: {food_data.shape[0]} compounds")

            # Extract SMILES
            column_names = food_data.columns.tolist()
            smiles_candidates = ['SMILES', 'smiles', 'Smiles']
            food_smiles_column = None

            for col in smiles_candidates:
                if col in column_names:
                    food_smiles_column = col
                    break

            if not food_smiles_column:
                food_smiles_column = column_names[0]

            self.food_smiles = food_data[food_smiles_column].values

            # Generate descriptors
            X_food = self.generate_descriptors_from_smiles(self.food_smiles)
            print(f"Screening data shape: {X_food.shape}")

            # Screening
            X_food_scaled = self.scaler.transform(X_food)
            predictions = self.best_model.predict(X_food_scaled)

            if hasattr(self.best_model, 'predict_proba'):
                probabilities = self.best_model.predict_proba(X_food_scaled)[:, 1]
            else:
                probabilities = np.zeros(len(predictions))

            # Create results dataframe
            results_df = pd.DataFrame({
                'Rank': range(1, len(predictions) + 1),
                'SMILES': self.food_smiles,
                'Prediction': predictions,
                'Predicted_Class': ['Active' if p == 1 else 'Inactive' for p in predictions],
                'Probability_Active': probabilities,
                'Probability_Inactive': 1 - probabilities
            })

            # Sort by probability
            results_df = results_df.sort_values('Probability_Active', ascending=False)
            results_df['Rank'] = range(1, len(results_df) + 1)

            return results_df

        except Exception as e:
            print(f"Error screening food compounds: {str(e)}")
            return None

    def save_all_results(self, food_results):
        """Save all results"""
        print(f"\n{'=' * 80}")
        print("💾 Saving All Results")
        print(f"{'=' * 80}")

        import joblib

        # 1. Save model comparison results
        comparison_data = []
        for name, metrics in self.results.items():
            row = {'Model': name, 'Description': self.models[name]['description']}
            row.update({k: v for k, v in metrics.items() if k not in ['confusion_matrix', 'roc_curve']})
            comparison_data.append(row)

        df_comparison = pd.DataFrame(comparison_data)
        df_comparison.to_csv('model_comparison_detailed.csv', index=False, encoding='utf-8-sig')
        print("✅ Detailed model comparison saved as 'model_comparison_detailed.csv'")

        # 2. Save best model
        joblib.dump({
            'model': self.best_model,
            'scaler': self.scaler,
            'model_name': self.best_model_name,
            'metrics': self.results[self.best_model_name],
            'feature_names': self.feature_names
        }, 'best_anxa1_model.pkl')
        print("✅ Best model saved as 'best_anxa1_model.pkl'")

        # 3. Save screening results
        if food_results is not None:
            # 修改：更新文件名，但保持功能不变
            food_results.to_csv('database_screening_results.csv', index=False, encoding='utf-8-sig')
            print("✅ Screening results saved as 'database_screening_results.csv'")

            # Save top 100 compounds
            top_100 = food_results.head(100)
            top_100.to_csv('database_screening_top_100.csv', index=False, encoding='utf-8-sig')
            print("✅ Top 100 compounds saved as 'database_screening_top_100.csv'")

            # Analyze flavonoid compounds
            flavonoid_count = self.analyze_flavonoids_in_results(food_results, top_n=50)

            # Generate screening results chart
            self.create_screening_results_chart(food_results, flavonoid_count)

    def create_screening_results_chart(self, food_results, flavonoid_count):
        """Create screening results statistical chart"""
        if food_results is None:
            return

        # 修改：增加图形大小和调整布局
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 修改：增加子图间距
        plt.subplots_adjust(
            wspace=0.3,
            hspace=0.4,
            left=0.08,
            right=0.95,
            top=0.92,
            bottom=0.08
        )

        # 1. Active/Inactive distribution
        ax1 = axes[0, 0]
        active_count = sum(food_results['Prediction'] == 1)
        inactive_count = sum(food_results['Prediction'] == 0)

        labels1 = ['Active', 'Inactive']
        sizes1 = [active_count, inactive_count]
        colors1 = ['#ff6b6b', '#4ecdc4']

        ax1.pie(sizes1, labels=labels1, colors=colors1, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Active Compound Distribution', fontsize=12, pad=20)

        # 2. Flavonoid distribution (top 50)
        ax2 = axes[0, 1]
        labels2 = ['Flavonoid', 'Non-Flavonoid']
        sizes2 = [flavonoid_count, 50 - flavonoid_count]
        colors2 = ['#ffd166', '#06d6a0']

        ax2.pie(sizes2, labels=labels2, colors=colors2, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Flavonoid Distribution in Top 50 Compounds', fontsize=12, pad=20)

        # 3. Probability distribution histogram
        ax3 = axes[1, 0]
        if 'Probability_Active' in food_results.columns:
            ax3.hist(food_results['Probability_Active'], bins=50, alpha=0.7, color='#118ab2')
            ax3.set_xlabel('Active Probability', fontsize=11, labelpad=10)
            ax3.set_ylabel('Number of Compounds', fontsize=11, labelpad=10)
            ax3.set_title('Active Probability Distribution', fontsize=12, pad=20)
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis='both', labelsize=9)

        # 4. Rank vs Probability relationship
        ax4 = axes[1, 1]
        top_50 = food_results.head(50)
        if len(top_50) > 0 and 'Probability_Active' in top_50.columns:
            ax4.scatter(top_50['Rank'], top_50['Probability_Active'], alpha=0.6, color='#ef476f')
            ax4.set_xlabel('Rank', fontsize=11, labelpad=10)
            ax4.set_ylabel('Active Probability', fontsize=11, labelpad=10)
            ax4.set_title('Rank vs Probability (Top 50 Compounds)', fontsize=12, pad=20)
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis='both', labelsize=9)

        # 修改：调整主标题位置
        plt.suptitle('Screening Results Statistical Analysis',
                     fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('screening_results_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Screening results analysis chart saved as 'screening_results_analysis.png'")


def main():
    """Main function"""
    print("=" * 100)
    print("🧬 ANXA1 Flavonoid Virtual Screening System")
    print("=" * 100)

    screener = ANXA1VirtualScreening()

    try:
        # 1. Load and prepare data
        X, y = screener.load_training_data()

        # 2. Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 3. Standardization
        X_train_scaled = screener.scaler.fit_transform(X_train)
        X_test_scaled = screener.scaler.transform(X_test)

        # 4. Initialize models
        screener.initialize_models()

        # 5. Train and compare models
        screener.train_and_compare_models(X_train_scaled, X_test_scaled, y_train, y_test)

        # 6. Display comparison table
        screener.display_model_comparison_table()

        # 7. Select best model
        screener.select_best_model()

        # 8. Create individual model charts
        screener.create_individual_model_charts()

        # 9. Create comparison visualizations
        screener.create_comparison_visualizations()

        # 10. Screen food compounds
        food_results = screener.screen_food_compounds()

        # 11. Save all results
        screener.save_all_results(food_results)

        print(f"\n{'=' * 100}")
        print("✅ Virtual Screening Completed!")
        print(f"{'=' * 100}")

        # Print summary
        print(f"\n📊 Final Results Summary:")
        print(f"  Best Model: {screener.best_model_name}")
        print(f"  F1 Score: {screener.results[screener.best_model_name].get('F1_Score', 0):.4f}")
        print(f"  AUC-ROC: {screener.results[screener.best_model_name].get('AUC_ROC', 0.5):.4f}")

        if food_results is not None:
            active_count = sum(food_results['Prediction'] == 1)
            print(f"  Total compounds screened: {len(food_results):,}")
            print(f"  Predicted active compounds: {active_count:,} ({active_count / len(food_results) * 100:.1f}%)")

        print(f"\n📁 Generated Files:")
        print("  Visualization Charts:")
        print("  - Individual model performance charts: SVM_performance.png, RF_performance.png, etc.")
        print("  - model_performance_radar.png (Performance radar chart)")
        print("  - model_roc_comparison.png (ROC curves comparison)")
        print("  - model_performance_comparison.png (Performance metrics comparison)")
        print("  - model_training_time_comparison.png (Training time comparison)")
        print("  - flavonoid_distribution.png (Flavonoid distribution)")
        print("  - screening_results_analysis.png (Screening results analysis)")
        print("  Data Files:")
        print("  - model_comparison_table.csv")
        print("  - model_comparison_detailed.csv")
        print("  - best_anxa1_model.pkl")
        print("  - database_screening_results.csv")  # 修改：更新输出文件名
        print("  - database_screening_top_100.csv")  # 修改：更新输出文件名
        print("  - flavonoid_compounds_top_50.csv")

    except Exception as e:
        print(f"\n❌ Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()