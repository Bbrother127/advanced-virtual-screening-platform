import os
import sys
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

# 设置上传目录
if 'PYTHONANYWHERE' in os.environ:
    # PythonAnywhere 环境
    app.config['UPLOAD_FOLDER'] = '/home/您的用户名/mysite/static/uploads'
else:
    # 本地环境
    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# RDKit 检查
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem import rdMolDescriptors
    RDKIT_AVAILABLE = True
except:
    RDKIT_AVAILABLE = False
    print("RDKit 不可用，使用简化模式")

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
        """生成分子描述符"""
        if not RDKIT_AVAILABLE:
            # RDKit 不可用时的替代方案
            descriptors = []
            for smiles in smiles_list:
                # 创建简单的模拟描述符
                mol_weight = len(smiles) * 10  # 模拟分子量
                num_atoms = smiles.count('C') + smiles.count('N') + smiles.count('O')  # 简单原子计数
                num_bonds = smiles.count('.') + smiles.count('(') + smiles.count(')')  # 简单键计数
                logp = mol_weight * 0.1  # 模拟 logP
                descriptors.append([mol_weight, num_atoms, num_bonds, logp])
            return np.array(descriptors)
        
        # 使用 RDKit 生成描述符
        descriptors = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    tpsa = Descriptors.TPSA(mol)
                    num_atoms = mol.GetNumAtoms()
                    descriptors.append([mw, logp, tpsa, num_atoms])
                else:
                    descriptors.append([0, 0, 0, 0])
            except:
                descriptors.append([0, 0, 0, 0])
        
        return np.array(descriptors)
    
    def is_flavonoid(self, smiles):
        """判断是否为黄酮类化合物"""
        if not RDKIT_AVAILABLE:
            # 简化判断：包含特定模式
            flav_patterns = ['C1=CC=CC=C1C', 'C=C[C@@H]', 'C=CC=CC=C1']
            return any(pattern in smiles for pattern in flav_patterns)
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            # 使用 RDKit 进行黄酮类判断
            # 这里简化处理，实际应该有更复杂的判断逻辑
            return 'C1=CC=CC=C1C' in smiles
        except:
            return False
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """训练和评估模型"""
        model_configs = {
            'SVM': {'model': SVC(probability=True, random_state=42), 'color': '#ff6b6b'},
            '逻辑回归': {'model': LogisticRegression(random_state=42), 'color': '#4ecdc4'},
            'KNN': {'model': KNeighborsClassifier(n_neighbors=5), 'color': '#45b7d1'},
            '随机森林': {'model': RandomForestClassifier(n_estimators=100, random_state=42), 'color': '#96ceb4'},
            'PLS-DA': {'model': LinearDiscriminantAnalysis(), 'color': '#ffeaa7'},
            'DFNN': {'model': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42), 'color': '#dda0dd'}
        }
        
        self.results = {}
        
        for name, info in model_configs.items():
            try:
                print(f"训练 {name} 模型...")
                model = info['model']
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else np.zeros(len(y_pred))
                
                # 计算指标
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                auc_roc = roc_auc_score(y_test, y_proba)
                
                # ROC 曲线
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                
                self.results[name] = {
                    'Accuracy': round(accuracy, 4),
                    'Precision': round(precision, 4),
                    'Recall': round(recall, 4),
                    'F1_Score': round(f1, 4),
                    'AUC_ROC': round(auc_roc, 4),
                    'roc_curve': {'fpr': fpr, 'tpr': tpr},
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                }
                
                self.models[name] = info
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                self.results[name] = {'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1_Score': 0, 'AUC_ROC': 0.5}
        
        # 选择最佳模型
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
        """筛选数据库"""
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
                    'smiles': smiles,
                    'prediction': int(pred),
                    'probability': float(prob)
                })
        
        # 统计信息
        stats = {
            'total': len(results),
            'active': sum(1 for r in results if r['prediction'] == 1),
            'inactive': sum(1 for r in results if r['prediction'] == 0),
            'flavonoid_top50': sum(1 for r in results[:50] if r['is_flavonoid']),
            'probability_bins': self._create_probability_bins(results),
            'flavonoid_compounds': flavonoid_compounds
        }
        
        self.screening_results = results
        return results, stats
    
    def _create_probability_bins(self, results, bins=10):
        """创建概率分布"""
        if not results:
            return []
        
        probabilities = [r['probability'] for r in results]
        min_prob, max_prob = min(probabilities), max(probabilities)
        bin_width = (max_prob - min_prob) / bins
        
        bins_data = []
        for i in range(bins):
            start = min_prob + i * bin_width
            end = min_prob + (i + 1) * bin_width
            count = sum(1 for p in probabilities if start <= p < end)
            bins_data.append({
                'range': f"{start:.3f}-{end:.3f}",
                'count': count
            })
        
        return bins_data

# 全局实例
screening_system = VirtualScreeningSystem()

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/upload-training-data', methods=['POST'])
def upload_training_data():
    """上传训练数据"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '请选择文件'})
        
        file = request.files['file']
        if file.filename == '' or not file.filename.endswith('.csv'):
            return jsonify({'success': False, 'error': '请上传CSV文件'})
        
        # 保存文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 读取数据
        df = pd.read_csv(filepath)
        
        # 确定SMILES列和标签列
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
        y = df[label_col].values
        
        # 处理标签
        if hasattr(y, 'dtype') and y.dtype == 'object':
            label_map = {}
            for val in np.unique(y):
                val_lower = str(val).lower()
                if val_lower in ['active', '1', 'yes', 'y', 'true', '+']:
                    label_map[val] = 1
                else:
                    label_map[val] = 0
            y = np.array([label_map[val] for val in y])
        
        # 生成特征
        X = screening_system.generate_descriptors(smiles_list)
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train_scaled = screening_system.scaler.fit_transform(X_train)
        X_test_scaled = screening_system.scaler.transform(X_test)
        
        # 训练和评估模型
        results, best_model = screening_system.train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # 保存训练信息
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
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/upload-screening-database', methods=['POST'])
def upload_screening_database():
    """上传筛选数据库"""
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
    """下载文件"""
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
        
        if isinstance(csv_data, str):
            response = make_response(csv_data.encode('utf-8-sig'))
        else:
            response = make_response(csv_data.encode('utf-8-sig'))
        
        response.headers['Content-Type'] = 'text/csv; charset=utf-8-sig'
        response.headers['Content-Disposition'] = f'attachment; filename={filename}'
        return response
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/clear-session', methods=['POST'])
def clear_session():
    """清除会话"""
    session.clear()
    screening_system.__init__()
    return jsonify({'success': True})

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    # 本地开发服务器
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)