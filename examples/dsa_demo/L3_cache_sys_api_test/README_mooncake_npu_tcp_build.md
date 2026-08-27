# Mooncake NPU TCP + AscendDirect 隔离构建

该流程用于验证 Mooncake NPU wheel 能否在同一个 vLLM-Ascend worker 进程中同时
提供：

- `protocol=ascend`：保留社区基线的 HBM/NPU 数据面；
- `protocol=tcp`：让 DSA rank-local Host L2 通过网络写入 Mooncake DRAM pool。

> 2026-08-29 更新：原始 `mooncake-transfer-engine-npu==0.3.11.post1` 已通过
> Store-owned TCP 的 swapped L2 单 worker、NPU0↔1 和跨 NUMA NPU0↔6 门禁。
> DSA TCP 首版不依赖本页自编 wheel。本构建仅保留作 transport dispatch、上游
> 修复验证和未来 TCP-only wheel 评估，不应成为默认部署前置步骤。

默认固定源码提交：

```text
e9c61075720039bcfc5fffd19f847608402be3d0
```

安装采用 `pip --target`，不会覆盖容器已有的
`mooncake-transfer-engine-npu==0.3.11.post1`。

## 首次构建

```bash
cd examples/dsa_demo/L3_cache_sys_api_test
bash build_install_mooncake_npu_tcp.sh bootstrap
```

`bootstrap` 会先安装 Mooncake 这条构建路径需要的系统开发包，再执行
`audit + build + install + verify-all`。支持 `apt-get`、`dnf` 和 `yum`，安装
系统包时需要 `root` 或 `sudo`。

脚本不会直接运行 Mooncake 上游较重的完整依赖脚本。当前关闭了 ETCD、Go、Rust、
benchmark 和 unit test，因此只安装 TCP、AscendDirect 和 Store 真正使用的依赖：

```text
gflags, glog, JsonCpp, ibverbs, numa, curl, zstd, xxhash
```

`yaml-cpp==0.8.0`、`msgpack-c==7.0.0` 和 Mooncake 固定提交自带的
`yalantinglibs` 会编译到私有目录，不写入 `/usr/local`。
三个旧依赖的独立 CMake 配置均显式设置
`CMAKE_POLICY_VERSION_MINIMUM=3.5`，可在 CMake 4.x 环境中构建。

默认目录：

```text
源码：$HOME/mooncake-dsa-npu-tcp-src
产物：$HOME/.cache/dsa-mooncake-npu-tcp/wheels
安装：$HOME/.cache/dsa-mooncake-npu-tcp/site
激活：$HOME/.cache/dsa-mooncake-npu-tcp/activate.sh
私有 C++ 依赖：$HOME/.cache/dsa-mooncake-npu-tcp/native
```

公司网络无法访问 GitHub 时，先准备 Mooncake 源码并指定：

```bash
MOONCAKE_SOURCE_DIR=/home/data/Mooncake \
bash build_install_mooncake_npu_tcp.sh bootstrap
```

脚本不会切换调用方管理的已有源码目录，只会打印当前提交。若由脚本自行 clone，
则会 checkout 到固定 commit 并更新 submodule。

## 分阶段执行

```bash
bash build_install_mooncake_npu_tcp.sh audit
bash build_install_mooncake_npu_tcp.sh deps
bash build_install_mooncake_npu_tcp.sh check-deps
bash build_install_mooncake_npu_tcp.sh build
bash build_install_mooncake_npu_tcp.sh install
bash build_install_mooncake_npu_tcp.sh verify-tcp
bash build_install_mooncake_npu_tcp.sh verify-ascend
```

完成一次 `deps` 后，后续重编可直接运行：

```bash
bash build_install_mooncake_npu_tcp.sh all
```

`build` 和 `all` 会在正式编译前执行一个很小的 CMake dependency probe。缺少
依赖时会在几十秒内失败并提示运行 `deps`，不会等到 Mooncake 大工程编译中途才报错。
该探针不仅查找头文件和 CMake target，还会实际编译、链接一个最小程序，以覆盖
私有静态库搜索路径等只在 link 阶段暴露的问题。
默认 `MOONCAKE_CLEAN_BUILD=1`，因此上一次失败留下的 Mooncake CMake 目录会自动
清理；旧的 `--system-site-packages` 构建 venv 也会被识别并重建，不需要手工删除。

构建 Python venv 不继承系统 site-packages，并将 `setuptools` 约束在
`>=77,<81`。因此系统环境中的 vLLM 依赖约束和实际使用的
`torch-npu==2.10.0.post6` 不会再干扰 wheel 构建；后者是运行环境选择，并不是本次
`yaml-cpp` 配置失败的原因。
构建阶段也会屏蔽 CANN 注入的外部 `PYTHONPATH`；运行时验证不会屏蔽 CANN 环境。
少数镜像会通过 Python 自身的 site 配置暴露 `ms-service-profiler` 元数据，pip 仍可能
打印其可选依赖缺失警告。只要构建工具最终显示 `Successfully installed`，该警告与
Mooncake wheel 构建无关，无需补装 profiler 的整套依赖。

### APT 无法创建临时文件

若日志包含：

```text
Couldn't create temporary file /tmp/apt.conf.* for passing config to apt-key
```

问题位于容器 `/tmp`，不是仓库签名或 Mooncake。`bootstrap/deps` 会检查并在可能时
把 `/tmp` 权限修复为标准的 `1777`，同时验证 `root` 和 `_apt` 用户均可创建文件。
如果文件系统只读、空间耗尽或 inode 耗尽，脚本会打印 `ls/df/mount` 诊断并停止，
不会再使用旧 APT 索引继续安装。

构建完成后激活当前 shell：

```bash
source "$HOME/.cache/dsa-mooncake-npu-tcp/activate.sh"

python - <<'PY'
import mooncake.engine
import mooncake.store
print(mooncake.engine.__file__)
print(mooncake.store.__file__)
PY
```

路径必须位于 `$HOME/.cache/dsa-mooncake-npu-tcp/site`。取消使用时退出当前 shell，
或恢复原来的 `PYTHONPATH`，系统 wheel 从未被覆盖。

## 真实 TCP 数据面

先启动 Mooncake Master：

```bash
bash baseline_e2e/start_mooncake_master.sh
```

另一个已经激活自定义 wheel 的终端先测普通 pinned Host：

```bash
ASCEND_VISIBLE_DEVICES=0,1 \
ASCEND_RT_VISIBLE_DEVICES=0,1 \
MOONCAKE_TP_ACTIVE_DEVICES=0,1 \
MOONCAKE_TP_COMM_MODE=tcp \
MOONCAKE_TP_BUFFER_MODE=pinned \
MOONCAKE_TP_PRODUCER_RANK=0 \
MOONCAKE_TP_CONSUMER_DEVICE=0 \
MOONCAKE_TP_TARGET_DEVICE=1 \
MOONCAKE_TP_IO_THREAD=background \
bash run_l3_cache_api_tests.sh mooncake-tp
```

通过后仅把 buffer 改成正式 DSA 地址：

```bash
ASCEND_VISIBLE_DEVICES=0,1 \
ASCEND_RT_VISIBLE_DEVICES=0,1 \
MOONCAKE_TP_ACTIVE_DEVICES=0,1 \
MOONCAKE_TP_COMM_MODE=tcp \
MOONCAKE_TP_BUFFER_MODE=swapped \
MOONCAKE_TP_PRODUCER_RANK=0 \
MOONCAKE_TP_CONSUMER_DEVICE=0 \
MOONCAKE_TP_TARGET_DEVICE=1 \
MOONCAKE_TP_IO_THREAD=background \
bash run_l3_cache_api_tests.sh mooncake-tp
```

验收日志必须满足：

1. `engine.initialize` 明确使用 `tcp`，不出现 `install AscendDirectTransport`；
2. 对象被强制定向到另一个 worker 的 segment；
3. put、remote get 和逐字节比较全部通过；
4. 最终打印 `[mooncake-tp] PASSED`。

该门禁通过后，DSA connector 仍保持社区基线的 worker 内嵌 Client 形态，只需
把 DSA L2 backend transport 设为 TCP。L3 loadback 第一版采用
`L3 -> local L2 -> HBM`，不要求 TCP 直接写 NPU HBM。
