import os
import numpy as np
import pandas as pd
import json
import joblib
from flask import Flask, render_template, request, jsonify, send_file, make_response
from flask import session
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, average_precision_score)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'anxa1_virtual_screening_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem import rdMolDescriptors
    RDKIT_AVAILABLE = True
except:
    RDKIT_AVAILABLE = False

class VirtualScreeningSystem:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.training_data = None
        self.screening_results = None
        self.flavonoid_results = None
        
    def generate_descriptors(self, smiles_list):
        if RDKIT_AVAILABLE:
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
            self.feature_names = ['分子量(MW)', 'LogP', '氢键受体(HBA)', '氢键供体(HBD)', '可旋转键', 
                                   'TPSA', '芳香环', '脂肪环', '总环数', 'Fsp3']
        else:
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
                    ]
                    descriptors_list.append(desc)
                else:
                    descriptors_list.append([0] * 6)
            self.feature_names = ['SMILES长度', 'C原子', 'O原子', 'N原子', '分支', '双键']
        return np.array(descriptors_list)
    
    def initialize_models(self):
        self.models = {
            'SVM': {
                'model': SVC(probability=True, random_state=42, class_weight='balanced'),
                'description': '支持向量机',
                'color': '#1f77b4'
            },
            'LR': {
                'model': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
                'description': '逻辑回归',
                'color': '#ff7f0e'
            },
            'KNN': {
                'model': KNeighborsClassifier(n_neighbors=5, weights='distance'),
                'description': 'K近邻',
                'color': '#2ca02c'
            },
            'RF': {
                'model': RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'),
                'description': '随机森林',
                'color': '#d62728'
            },
            'PLS-DA': {
                'model': LinearDiscriminantAnalysis(),
                'description': '偏最小二乘',
                'color': '#9467bd'
            },
            'DFNN': {
                'model': MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42, max_iter=1000, early_stopping=True, validation_fraction=0.1),
                'description': '深度神经网络',
                'color': '#8c564b'
            }
        }
    
    def is_flavonoid(self, smiles):
        if not isinstance(smiles, str):
            return False
        smiles_lower = smiles.lower()
        score = 0
        
        flavonoid_patterns = [
            'c1ccc2c(c1)c(=o)cc(o2)',
            'c1cc2c(cc1)oc(=o)cc2',
        ]
        
        for pattern in flavonoid_patterns:
            if pattern in smiles_lower:
                score += 3
                break
        
        oh_count = smiles_lower.count('o') - smiles_lower.count('oc') * 0.5
        if oh_count >= 2: score += 1
        if oh_count >= 3: score += 1
        
        ring_count = sum(1 for char in smiles if char.isdigit())
        if ring_count >= 2: score += 1
        if ring_count >= 3: score += 1
        
        if 'oc' in smiles_lower: score += 0.5
        if 'n' in smiles_lower: score += 0.5
        
        return score >= 2.5
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        self.initialize_models()
        import time
        
        for name, info in self.models.items():
            try:
                start_time = time.time()
                info['model'].fit(X_train, y_train)
                train_time = time.time() - start_time
                
                start_time = time.time()
                y_pred = info['model'].predict(X_test)
                predict_time = time.time() - start_time
                
                metrics = {
                    'Accuracy': round(accuracy_score(y_test, y_pred), 4),
                    'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
                    'Recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
                    'F1_Score': round(f1_score(y_test, y_pred, zero_division=0), 4),
                    'Training_Time(s)': round(train_time, 4),
                    'Prediction_Time(s)': round(predict_time, 4),
                }
                
                if hasattr(info['model'], 'predict_proba'):
                    y_proba = info['model'].predict_proba(X_test)[:, 1]
                    if len(np.unique(y_test)) > 1:
                        metrics['AUC_ROC'] = round(roc_auc_score(y_test, y_proba), 4)
                        metrics['AUPRC'] = round(average_precision_score(y_test, y_proba), 4)
                        
                        fpr, tpr, _ = roc_curve(y_test, y_proba)
                        metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
                else:
                    metrics['AUC_ROC'] = 0.5
                    metrics['AUPRC'] = 0.0
                
                cm = confusion_matrix(y_test, y_pred)
                metrics['confusion_matrix'] = cm.tolist()
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    metrics['TN'] = int(tn)
                    metrics['FP'] = int(fp)
                    metrics['FN'] = int(fn)
                    metrics['TP'] = int(tp)
                    metrics['Specificity'] = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0
                
                cv_scores = cross_val_score(info['model'], X_train, y_train, cv=5, scoring='f1')
                metrics['CV_F1_Mean'] = round(cv_scores.mean(), 4)
                metrics['CV_F1_Std'] = round(cv_scores.std(), 4)
                
                self.results[name] = metrics
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                self.results[name] = {'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1_Score': 0, 'AUC_ROC': 0.5}
        
        best_score = -1
        for name, metrics in self.results.items():
            auc = metrics.get('AUC_ROC', 0.5)
            f1 = metrics.get('F1_Score', 0)
            score = 0.5 * auc + 0.5 * f1
            if score > best_score:
                best_score = score
                self.best_model_name = name
                self.best_model = self.models[name]['model']
        
        return self.results, self.best_model_name
    
    def screen_database(self, smiles_list):
        X = self.generate_descriptors(smiles_list)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.best_model.predict(X_scaled)
        if hasattr(self.best_model, 'predict_proba'):
            probabilities = self.best_model.predict_proba(X_scaled)[:, 1]
        else:
            probabilities = np.zeros(len(predictions))
        
        results = []
        flavonoid_compounds = []
        
        for i, (smiles, pred, prob) in enumerate(zip(smiles_list, predictions, probabilities)):
            is_flav = self.is_flavonoid(smiles)
            result = {
                'rank': i + 1,
                'smiles': smiles,
                'prediction': int(pred),
                'probability': float(prob),
                'is_flavonoid': is_flav
            }
            results.append(result)
            
            if is_flav and i < 50:
                flavonoid_compounds.append({
                    'rank': i + 1,
                    'smiles': smiles[:50] + '...' if len(smiles) > 50 else smiles,
                    'probability': round(float(prob), 4),
                    'flavonoid': 'Yes'
                })
        
        results = sorted(results, key=lambda x: x['probability'], reverse=True)
        for i, r in enumerate(results):
            r['rank'] = i + 1
        
        screening_system.screening_results = results
        
        active_count = sum(1 for r in results if r['prediction'] == 1)
        inactive_count = len(results) - active_count
        flavonoid_count = sum(1 for r in results[:50] if r['is_flavonoid'])
        
        probability_bins = [0] * 10
        for r in results:
            bin_idx = min(int(r['probability'] * 10), 9)
            probability_bins[bin_idx] += 1
        
        return results, {
            'total': len(smiles_list),
            'active': active_count,
            'inactive': inactive_count,
            'flavonoid_top50': flavonoid_count,
            'probability_bins': probability_bins,
            'flavonoid_compounds': flavonoid_compounds[:50]
        }

screening_system = VirtualScreeningSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload-training-data', methods=['POST'])
def upload_training_data():
    try:
        file = request.files.get('file')
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(filepath)
            df = pd.read_csv(filepath)
            
            smiles_col = None
            for col in ['SMILES', 'smiles', 'Smiles']:
                if col in df.columns:
                    smiles_col = col
                    break
            if not smiles_col:
                smiles_col = df.columns[0]
            
            label_col = None
            for col in ['Labels', 'labels', 'Label', 'label']:
                if col in df.columns:
                    label_col = col
                    break
            if not label_col:
                label_col = df.columns[-1]
            
            smiles_list = df[smiles_col].values.tolist()
            X = screening_system.generate_descriptors(smiles_list)
            y = df[label_col].values
            
            if hasattr(y, 'dtype') and y.dtype == 'object':
                label_map = {}
                for val in np.unique(y):
                    val_lower = str(val).lower()
                    if val_lower in ['active', '1', 'yes', 'y', 'true', '+']:
                        label_map[val] = 1
                    else:
                        label_map[val] = 0
                y = np.array([label_map[val] for val in y])
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            X_train_scaled = screening_system.scaler.fit_transform(X_train)
            X_test_scaled = screening_system.scaler.transform(X_test)
            
            results, best_model = screening_system.train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test)
            
            session['model_trained'] = True
            session['training_info'] = {
                'total_samples': len(df),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'active_samples': int(sum(y == 1)),
                'inactive_samples': int(sum(y == 0)),
                'features': screening_system.feature_names
            }
            
            return jsonify({
                'success': True,
                'data_info': session['training_info'],
                'model_results': results,
                'best_model': best_model,
                'model_colors': {name: info['color'] for name, info in screening_system.models.items()}
            })
        
        return jsonify({'success': False, 'error': '请上传CSV文件'})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/upload-screening-database', methods=['POST'])
def upload_screening_database():
    try:
        if not session.get('model_trained'):
            return jsonify({'success': False, 'error': '请先训练模型'})
        
        file = request.files.get('file')
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(filepath)
            df = pd.read_csv(filepath)
            
            smiles_col = None
            for col in ['SMILES', 'smiles', 'Smiles']:
                if col in df.columns:
                    smiles_col = col
                    break
            if not smiles_col:
                smiles_col = df.columns[0]
            
            smiles_list = df[smiles_col].values.tolist()
            
            results, stats = screening_system.screen_database(smiles_list)
            
            session['screening_info'] = stats
            
            return jsonify({
                'success': True,
                'screening_results': {
                    'total_compounds': stats['total'],
                    'active_compounds': stats['active'],
                    'inactive_compounds': stats['inactive'],
                    'flavonoid_in_top50': stats['flavonoid_top50'],
                    'probability_bins': stats['probability_bins'],
                    'top_results': results[:100],
                    'flavonoid_compounds': stats['flavonoid_compounds']
                }
            })
        
        return jsonify({'success': False, 'error': '请上传CSV文件'})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/download-file', methods=['POST'])
def download_file():
    try:
        data = request.get_json()
        filename = data.get('filename', '')
        
        if filename == 'model_comparison_table.csv':
            comparison_data = []
            for name, metrics in screening_system.results.items():
                row = {
                    '模型': name,
                    '准确率': metrics.get('Accuracy', 0),
                    '精确率': metrics.get('Precision', 0),
                    '召回率': metrics.get('Recall', 0),
                    'F1分数': metrics.get('F1_Score', 0),
                    'AUC_ROC': metrics.get('AUC_ROC', 0.5),
                    '特异性': metrics.get('Specificity', 0),
                    'CV_F1均值': metrics.get('CV_F1_Mean', 0),
                    '训练时间(s)': metrics.get('Training_Time(s)', 0),
                }
                comparison_data.append(row)
            df = pd.DataFrame(comparison_data)
            df = df.sort_values('F1分数', ascending=False)
            csv_data = df.to_csv(index=False, encoding='utf-8-sig')
        elif filename == 'model_comparison_detailed.csv':
            detailed_data = []
            for name, metrics in screening_system.results.items():
                row = {'模型': name}
                for k, v in metrics.items():
                    if k not in ['roc_curve', 'confusion_matrix']:
                        row[k] = v
                detailed_data.append(row)
            csv_data = pd.DataFrame(detailed_data).to_csv(index=False, encoding='utf-8-sig')
        elif filename == 'database_screening_results.csv' and hasattr(screening_system, 'screening_results') and screening_system.screening_results:
            sr = screening_system.screening_results
            sr_df = pd.DataFrame([{
                '排名': r['rank'],
                'SMILES': r['smiles'],
                '预测结果': '活性' if r['prediction'] == 1 else '非活性',
                '活性概率': round(r['probability'], 4),
                '非活性概率': round(1 - r['probability'], 4),
                '黄酮类': '是' if r['is_flavonoid'] else '否'
            } for r in sr])
            csv_data = sr_df.to_csv(index=False, encoding='utf-8-sig')
        elif filename == 'database_screening_top_100.csv' and hasattr(screening_system, 'screening_results') and screening_system.screening_results:
            sr = screening_system.screening_results[:100]
            sr_df = pd.DataFrame([{
                '排名': r['rank'],
                'SMILES': r['smiles'],
                '预测结果': '活性' if r['prediction'] == 1 else '非活性',
                '活性概率': round(r['probability'], 4),
                '黄酮类': '是' if r['is_flavonoid'] else '否'
            } for r in sr])
            csv_data = sr_df.to_csv(index=False, encoding='utf-8-sig')
        elif filename == 'flavonoid_compounds_top_50.csv' and session.get('screening_info'):
            fc = session.get('screening_info', {}).get('flavonoid_compounds', [])
            if fc:
                csv_data = pd.DataFrame(fc).to_csv(index=False, encoding='utf-8-sig')
            else:
                csv_data = 'No flavonoid compounds found'
        else:
            csv_data = 'File not available'
        
        response = make_response(csv_data.encode('utf-8-sig'))
        response.headers['Content-Type'] = 'text/csv; charset=utf-8-sig'
        response.headers['Content-Disposition'] = f'attachment; filename={filename}'
        return response
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/clear-session', methods=['POST'])
def clear_session():
    session.clear()
    screening_system.__init__()
    return jsonify({'success': True})

@app.route('/health', methods=['GET'])
def health_check():
    # Simple endpoint to verify server is responsive
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    # Development server
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)
else:
    # Production server configuration for Render
    import os
    from werkzeug.middleware.proxy_fix import ProxyFix
    
    # Apply proxy fix for Render
    app.wsgi_app = ProxyFix(app.wsgi_app)
    
    # Set host to 0.0.0.0 for external access
    host = '0.0.0.0'
    
    # Get port from environment variable (Render uses PORT)
    port = int(os.environ.get('PORT', 5000))
    
    # Run the Flask app
    app.run(host=host, port=port, debug=False)
