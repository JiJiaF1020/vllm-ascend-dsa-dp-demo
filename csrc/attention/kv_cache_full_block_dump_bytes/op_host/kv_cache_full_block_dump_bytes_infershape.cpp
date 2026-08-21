/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include <cstddef>

#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferKvCacheFullBlockDumpBytesShape(
    gert::InferShapeContext* context)
{
    if (context == nullptr || context->GetInputShape(1) == nullptr ||
        context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = *context->GetInputShape(1);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferKvCacheFullBlockDumpBytesDataType(
    gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_INT8);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(KvCacheFullBlockDumpBytes)
    .InferShape(InferKvCacheFullBlockDumpBytesShape)
    .InferDataType(InferKvCacheFullBlockDumpBytesDataType);
}  // namespace ops
