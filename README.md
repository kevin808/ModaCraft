# ModaCraft

ModaCraft 是一个多模态内容分析助手，能够集成搜索、分析和内容生成功能，为用户提供全面的信息服务。

## 功能特点

- 智能搜索：通过 SearxNG 进行多源搜索
- 多模态分析：支持文本、图片等多种内容类型的处理
- 结构化输出：提供清晰的分析报告和参考源
- 实时可视化：基于 Streamlit 的交互式界面

## 技术栈

- Python
- Streamlit
- OpenAI API
- SearxNG
- Pydantic
- HTTPX

## 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/ModaCraft.git
cd ModaCraft
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置环境：
   - 创建 `config.py` 文件
   - 设置以下配置项：
     - API_KEY (OpenAI API密钥)
     - BASE_URL (API基础URL)
     - SEARXNG_URL (SearxNG实例URL)

4. 运行应用：
```bash
streamlit run app.py
```

## 使用说明

1. 在输入框中输入您的问题
2. 点击"分析"按钮
3. 等待系统处理并展示结果
   - 分析报告
   - 相关图片（如果有）
   - 参考来源

## 开发说明

项目使用 Pydantic 进行数据验证，采用异步编程处理 API 请求，并实现了请求重试机制。