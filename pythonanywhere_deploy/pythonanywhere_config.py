# PythonAnywhere Web 应用配置
# 文件路径: /home/brotherB/pythonanywhere_config.py

import sys

# 添加应用目录到 Python 路径
sys.path.insert(0, '/home/brotherB/mysite')

# 导入 Flask 应用
from app import app

# 创建 WSGI 应用
application = app