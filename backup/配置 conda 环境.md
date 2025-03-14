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
