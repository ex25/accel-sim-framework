# 本机调用 Modal H100

本机已登录，profile 是 `moe-h100`，workspace 是 `gcarsten777`。无需 SSH；`modal run` 会把指定函数交给远端 GPU 执行。

```bash
export PATH="/home/wangjz/.local/share/accel-sim-moe-modal/bin:$PATH"
modal profile list
# 仅在登录失效时重新授权，不需要复制或输出 token：
# modal token new --profile moe-h100
```

下面是最小示例，保存为 `h100_check.py`；将 `nvidia-smi` 换成自己的验证命令后运行：

```python
import subprocess
import modal

image = modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.04-py3")
app = modal.App("hopper-ptx-check")

@app.function(image=image, gpu="H100!", timeout=180,
              max_containers=1, retries=0, single_use_containers=True)
def check():
    return subprocess.check_output(["nvidia-smi"], text=True)

@app.local_entrypoint()
def main():
    print(check.remote())
```

```bash
modal run --profile moe-h100 h100_check.py
```

上传测试文件时，在 `@app.function` 之前添加：

```python
if modal.is_local():
    image = image.add_local_dir("/tmp/h100-case", "/work", copy=True)
```

远端即可用 `subprocess.run(["bash", "run.sh"], cwd="/work", check=True)` 执行该目录里的测试。先在本地准备好源码、PTX、输入和构建产物，再提交所需的小包。远端应使用原生 CUDA 库。

`H100!` 固定申请 H100，避免被自动升级为 H200；日志会显示 app 链接。调用完成即结束本次任务，不需要 `modal deploy`。本说明写入时只核对了本机 CLI/profile，没有新申请 GPU。[GPU 文档](https://modal.com/docs/guide/gpu) · [本地文件上传](https://modal.com/docs/guide/local-data)
