# 高级虚拟筛选平台：机器学习驱动的药物发现引擎

一个基于机器学习的虚拟筛选平台，用于药物发现和分子活性预测。

## 功能特性

- **6种机器学习模型**：SVM、逻辑回归、KNN、随机森林、PLS-DA、深度前馈神经网络
- **12种可视化图表**：ROC曲线、混淆矩阵、特征重要性等
- **用户友好的Web界面**：现代化的设计，清晰的布局
- **文件上传功能**：支持SMILES格式文件上传
- **实时结果展示**：即时显示预测结果和可视化图表

## 技术栈

- **后端**：Python + Flask
- **机器学习**：scikit-learn, RDKit
- **数据可视化**：matplotlib, seaborn
- **前端**：HTML, CSS, JavaScript

## 部署

### 本地运行

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行应用：
```bash
python app.py
```

3. 访问：http://localhost:5000

### Render部署

1. Fork或克隆此仓库
2. 在Render上创建新服务
3. 选择Web服务类型
4. 配置：
   - Name: advanced-virtual-screening-platform
   - Repository: your-github-repo
   - Branch: main/master
   - Runtime: Python 3.11
   - Build Command: `bash build.sh`
   - Start Command: `gunicorn app:app`

## 使用说明

1. 访问平台主页
2. 上传包含SMILES格式的CSV文件
3. 选择机器学习模型
4. 查看预测结果和可视化图表
5. 下载结果报告

## 文件结构

```
├── app.py                 # Flask应用主文件
├── requirements.txt       # Python依赖
├── render.yaml           # Render部署配置
├── build.sh              # 构建脚本
├── static/               # 静态文件
│   ├── main.js           # 前端JavaScript
│   └── style.css         # 样式文件
└── templates/            # HTML模板
    └── index.html        # 主页面
```

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！