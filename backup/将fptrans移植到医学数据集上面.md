### 预处理医学数据集，封装进dataset、dataloader
#### 下载医疗数据集专门需要的软件包：dcm2niix
GitHub 早在 2021 年 8 月 就 取消了通过账号密码直接进行 HTTPS 验证的方式，不再支持用用户名 + 密码克隆私有仓库或推送代码。
所以推荐使用 SSH key 或者 GitHub 的个人访问令牌 (Personal Access Token, 简称 PAT)。

使用PAT（最后我使用的是conda）：

1. 先在github上找到软件包对应的网页
2. 再登录 GitHub → 右上角头像 → Settings → Developer Settings → Personal Access Tokens → Tokens (classic) → Generate new token。
3. 勾选 scopes，建议选择：repo → 访问私有仓库；read:org；workflow。
4. 设置一个有效期，比如 30 天或 90 天。
5. 生成后 复制 token。
6. 克隆时：`git clone https://<your-token>@github.com/rordenlab/dcm2niix.git`

不过，还要将dcm2nii加入PATH，太麻烦了。遂改为用conda：`conda install -c conda-forge dcm2niix](https://anaconda.org/conda-forge/dcm2niix`
#### 

### 修改细节，让整个程序可以运行通
