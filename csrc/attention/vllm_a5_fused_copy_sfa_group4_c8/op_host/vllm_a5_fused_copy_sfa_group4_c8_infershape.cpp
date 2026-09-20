#include "register/op_impl_registry.h"
namespace {
ge::graphStatus InferShape(gert::InferShapeContext *c)
{
    if (!c || !c->GetInputShape(0) || !c->GetInputShape(1) ||
        !c->GetOutputShape(0) || !c->GetOutputShape(1) ||
        c->GetInputShape(0)->GetDimNum() != 3) return ge::GRAPH_FAILED;
    *c->GetOutputShape(0) = *c->GetInputShape(1);
    *c->GetOutputShape(1) = *c->GetInputShape(0);
    c->GetOutputShape(1)->SetDim(2, 512);
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus InferType(gert::InferDataTypeContext *c)
{
    if (!c) return ge::GRAPH_FAILED;
    c->SetOutputDataType(0, ge::DT_INT8);
    c->SetOutputDataType(1, c->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}
IMPL_OP_INFERSHAPE(VllmA5FusedCopySfaGroup4C8).InferShape(InferShape).InferDataType(InferType);
