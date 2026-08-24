# MemCache A5 独立 IO 探针

该探针不依赖 vLLM/vLLM-Ascend，用于逐层验证容器中
`memcache_hybrid==1.2.0` 与 `memfabric_hybrid==1.2.0` 的公开 IO API。

脚本：`memcache_a5_probe.py`

推荐通过同目录的统一门禁入口执行，它会自动保存日志并保留经过 `tee` 后的
真实 Python 退出码：

```bash
bash run_l3_cache_api_tests.sh memcache-capabilities
bash run_l3_cache_api_tests.sh memcache-host
bash run_l3_cache_api_tests.sh memcache-npu
bash run_l3_cache_api_tests.sh memcache-gva
```

## 1. Mooncake 包名说明

`pip list` 中只有 `mooncake-transfer-engine-npu` 不表示缺少 Mooncake Store。
该 wheel 的构建脚本会同时打包：

- `mooncake.engine`；
- `mooncake.store`；
- `mooncake_master`；
- `mooncake_client`。

可在服务器检查：

```bash
python - <<'PY'
import mooncake
import mooncake.engine
import mooncake.store
import shutil

print("mooncake:", mooncake.__file__)
print("engine:", mooncake.engine.__file__)
print("store:", mooncake.store.__file__)
print("master:", shutil.which("mooncake_master"))
PY
```

## 2. 环境准备

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 实际安装路径可先用 pip show 查询；如果存在 set_env.sh，再 source。
pip show memcache_hybrid memfabric_hybrid
source /usr/local/memfabric_hybrid/set_env.sh
source /usr/local/memcache_hybrid/set_env.sh
```

确认没有旧服务占用探针默认端口：

```bash
ss -lntp | grep -E ':(5000|6000|18000)\b' || true
```

## 3. Gate 0：导入与 API

```bash
python memcache_a5_probe.py --mode capabilities
```

预期看到四个核心类，以及 `batch_put_from_layers`、
`batch_get_into_layers`、`register_buffer` 等 API 全部为 `True`。

## 4. Gate 1：Host DRAM 往返

该测试使用 `host_shm`，不验证 NPU/UBOE，只验证：

```text
MetaService + ConfigStore + LocalService
bytes object put/get
Host scatter-gather -> global DRAM -> Host scatter-gather
exists + bitwise readback
```

```bash
python memcache_a5_probe.py \
  --mode host \
  --protocol host_shm \
  --dram-size 1GB \
  --max-dram-size 4GB \
  --world-size 8
```

通过标志：

```text
[memcache-probe] HOST ROUND TRIP PASSED
[memcache-probe] ALL REQUESTED TESTS PASSED
```

## 5. Gate 2：A5 NPU 往返

先只暴露一张卡，降低环境变量与 local service 数量：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
```

按现场 A5 网络配置选择 `device_uboe` 或 `device_urma`。源码 v1.2.0
支持这两个 protocol，但 vLLM-Ascend v0.23 文档没有给出 A5 MemCache 完整
认证矩阵，因此失败时需要结合 MemFabric 日志继续定位。

`device_uboe/device_urma` 要求 `max_dram_size * world_size > 32TB`。下面只实际
分配 1GB DRAM，`256GB * 256` 是全局虚拟容量上限，不会一次分配 64TB：

```bash
python memcache_a5_probe.py \
  --mode npu \
  --protocol device_uboe \
  --device-id 0 \
  --dram-size 1GB \
  --max-dram-size 256GB \
  --world-size 256
```

如果现场使用 URMA：

```bash
python memcache_a5_probe.py \
  --mode npu \
  --protocol device_urma \
  --device-id 0 \
  --dram-size 1GB \
  --max-dram-size 256GB \
  --world-size 256
```

探针会显式验证：

```text
bytes object put/get
NPU HBM --COPY_L2G--> MemCache global DRAM
MemCache global DRAM --COPY_G2L--> NPU HBM
多 key、多 layer、非单一地址段
register/unregister buffer
bitwise readback
```

通过标志：

```text
[memcache-probe] NPU ROUND TRIP PASSED
```

## 6. Gate 3：vLLM-Ascend layerwise/GVA API

这一关覆盖 AscendStore MemCache layerwise 路径额外使用的 API：

```text
batch_alloc
batch_copy(L2G)
batch_write_finish
batch_get_key_info
batch_add_lease
batch_copy(G2L)
batch_remove_lease
```

```bash
python memcache_a5_probe.py \
  --mode gva \
  --protocol device_uboe \
  --device-id 0 \
  --dram-size 1GB \
  --max-dram-size 256GB \
  --world-size 256
```

也可以在前两关稳定后执行所有数据面：

```bash
python test_memcache.py \
  --mode all \
  --protocol device_uboe \
  --device-id 0 \
  --dram-size 1GB \
  --max-dram-size 256GB \
  --world-size 256
```

通过标志：

```text
[memcache-probe] GVA/LAYERWISE API ROUND TRIP PASSED
```

## 7. 使用已有 MetaService

若已经单独启动 MetaService：

```bash
python memcache_a5_probe.py \
  --mode host \
  --protocol host_shm \
  --external-meta
```

探针不会终止外部服务。

## 8. 结果反馈

失败时保留完整标准输出，并额外执行：

```bash
python - <<'PY'
import os
import memcache_hybrid
import memfabric_hybrid

print("memcache:", memcache_hybrid.__file__)
print("memfabric:", memfabric_hybrid.__file__)
for key in sorted(os.environ):
    if key.startswith(("ASCEND", "HCCL", "MMC", "MF", "PYTHON")):
        print(f"{key}={os.environ[key]}")
PY

npu-smi info
npu-smi info -t topo
```

判断顺序：

1. capabilities 失败：wheel/动态库/环境变量问题；
2. Host 失败：MetaService/ConfigStore/LocalService 问题；
3. Host 通过、NPU setup 失败：A5 protocol、HCOM/UBOE/URMA 配置问题；
4. NPU put/get 返回非零：buffer 注册、copy direction 或 global pool 状态问题；
5. 返回 0 但 bitwise 失败：真实 IO/同步/地址计算问题。

## 9. v1.2.0 清理行为

源码中的 `ResourceTracker` 是函数静态单例，但它在自身构造函数里又调用
`std::atexit(exitHandler)`。按照 native 退出回调的注册顺序，单例的
`mutex/unordered_set` 可能先析构，随后 `exitHandler` 再访问已经析构的对象，产生
`double free or corruption`。所以仅调用 `DistributedObjectStore.close()`、销毁
Python wrapper 和 GC 仍不足以规避该 v1.2.0 生命周期问题。

探针会先显式执行 `store.close()`，再销毁 wrapper；MetaService v1.2.0 只有阻塞的
`main()`，没有 Python stop API，因此在 LocalService 清理后用 SIGKILL 结束探针
自己创建的临时 MetaService。最后探针通过 `os._exit()` 跳过有缺陷的 native
全局 finalizer。数据面、显式 close 或校验失败仍会返回非零，只有已经完成显式
清理之后的 extension/global finalizer 被跳过。

若要专门复现上游 native 退出缺陷，可执行：

```bash
python test_memcache.py \
  --mode host \
  --protocol host_shm \
  --dram-size 1GB \
  --max-dram-size 4GB \
  --world-size 8 \
  --strict-native-teardown
```

该严格模式当前预期可能以退出码 `134` 结束，不应作为数据面门禁命令。
