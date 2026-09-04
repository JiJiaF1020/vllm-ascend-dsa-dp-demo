# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA 本地 DRAM 与池化 L3 缓存之间的任务规划和 IO 适配。

``schema`` 定义内容 key，``metadata`` 定义进程间任务，``scheduler`` 负责
命中与卸载规划，``worker`` 执行同步 load 和异步 store，``backend_adapter``
适配存储 API；``connector`` 将这些职责接入 vLLM 原生 KVConnector 生命周期。
稳定加载路径为 L3 -> 本 rank L2 -> HBM，卸载为 L2 -> L3 异步 write-behind。
是否允许 DP/MTP 等组合以 ``dsa_offload.config`` 的当前支持边界为准。
"""
