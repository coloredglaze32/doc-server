# 📃 **关于doc-server**

基于  [🦜️🔗 LangChain](https://github.com/hwchase17/langchain) 与  DeepSeek R1 大语言模型的本地知识库问答。

本项目是本地知识库问答应用的 serve 后端。目前实现基本的 RAG 功能。
后续会系统学习 langchain ，逐步添加更多的功能。

## 快速上手

```shell
# 打开 ubuntu 终端，切换 r1 环境
conda activate r1

# 打开目录
cd Project

# 拉取项目
$ git clone https://github.com/YuiGod/py-doc-qa-deepseek-server.git

# 进去项目
$ cd py-doc-qa-deepseek-server

# 安装项目相关依赖
pip install -r requirements.txt

# 进入 app 目录
cd app

# 启动服务
python main.py
```

## 项目功能

1. 文档管理API，文档上传到指定位置，并在 SQLite 记录信息。
2. 聊天对话历史管理API，用 SQLite 保存记录。
3. 聊天采用流式响应。
4. 实现基本的 RAG 功能。

> 基本框架已经搭建完成。后续会系统学习 LangGraph ，添加更多新的功能。

## src 目录树

```
    app                             # 主目录
    ├── core                        # LangChan 核心代码
    │   ├── base.py                 # LangChan 常量配置
    │   ├── langchain_retrieval.py  # 构建检索连
    │   └── langchain_vector.py     # 读取文档，分割文档，向量化文档
    ├── crud                        # 数据库 crud 操作目录
    │   ├── __init__.py
    │   ├── base.py                 # 数据库配置
    │   ├── chat_history_crud.py    # 对话聊天历史 crud
    │   ├── chat_session_crud.py    # 会话管理 crud
    │   └── document_crud.py        # 文档管理 crud
    ├── models                      # 数据库模型，基本模型目录
    │   ├── __init__.py
    │   ├── chat_history_model.py   # 聊天历史记录管理数据库模型
    │   ├── chat_model.py           # 聊天模型，基本模型
    │   ├── chat_session_model.py   # 会话管理数据库模型
    │   └── document_model.py       # 文档挂你数据库模型
    └── routers                     # api 路由分类
    │   ├── __init__.py
    │   ├── base.py                 # 基础配置，配置成功和失败返回模型
    │   ├── chat_router.py          # 聊天 Api
    │   ├── chat_session_router.py  # 会话管理 Api
    │   └── document_router.py      # 文档管理 Api
    ├── document_qa.db              # SQLite数据库
    ├── main.py                     # 主程序启动服务入口
```
