### 要避免 conda 和 pip 混用导致的版本冲突

#### pip 和 conda 管理包的方式不同

-   conda 是基于预编译的二进制文件，安装时会严格保持包的兼容性，不轻易更改版本。
-   pip 主要是源代码编译安装，可能会忽略现有 numpy 版本，直接安装最新或兼容的版本。

如果实在必须要用 pip 安装软件包，那么可以选择

1. 先用 conda 安装 numpy 等其他软件包，并用 ` pip install --no-deps ` 避免依赖冲突

```
conda install numpy=1.21.2
pip install dropblock pycocotools --no-deps
```

2.  冻结 numpy 等软件包的版本，防止 pip 自动更改：
```
echo "numpy==1.21.2" > requirements.txt
pip install -r requirements.txt
```

 ### 安装miniconda
跟着教程：
[https://blog.csdn.net/AlgoZZi/article/details/145074821](url)

如果 PowerShell 可以正常识别 conda 命令，但 VSCode 的终端无法识别，可能是由于 VSCode 的终端没有加载正确的环境变量。
重启 VSCode即可。

### 创建虚拟环境的全流程
```
conda create -n abag python=3.8 -y  # 创建虚拟环境
conda activate abag  # 激活环境
```
没有显示报错，但是无法激活环境：
有时 VSCode 会显示默认的 Python 环境而不是激活的 Conda 环境，确保在 VSCode 中选择了正确的 Python 解释器。

1.   按 Ctrl + Shift + P 打开命令面板
2.   输入并选择 Python: Select Interpreter
3.   在列表中选择你刚刚创建的 Conda 环境（myenv）

这将确保 VSCode 使用虚拟环境中的 Python 解释器。
