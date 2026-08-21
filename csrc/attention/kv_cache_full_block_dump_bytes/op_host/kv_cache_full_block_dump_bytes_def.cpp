/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include <vector>

#include "register/op_def_registry.h"

namespace ops {
class KvCacheFullBlockDumpBytes : public OpDef {
public:
    explicit KvCacheFullBlockDumpBytes(const char* name) : OpDef(name)
    {
        const std::vector<ge::DataType> bytes = {ge::DT_INT8};
        const std::vector<ge::DataType> ints = {ge::DT_INT32};
        const std::vector<ge::Format> formats = {ge::FORMAT_ND};
        this->Input("src_cache").ParamType(REQUIRED)
            .DataType(bytes).Format(formats);
        this->Input("dst_cache").ParamType(REQUIRED)
            .DataType(bytes).Format(formats);
        this->Input("src_block_ids").ParamType(REQUIRED)
            .DataType(ints).Format(formats);
        this->Input("dst_block_ids").ParamType(REQUIRED)
            .DataType(ints).Format(formats);
        this->Output("dst_cache").ParamType(REQUIRED)
            .DataType(bytes).Format(formats);
        this->AICore()
            .AddConfig("ascend910_93")
            .AddConfig("ascend910b")
            .AddConfig("ascend950");
    }
};

OP_ADD(KvCacheFullBlockDumpBytes);
}  // namespace ops
