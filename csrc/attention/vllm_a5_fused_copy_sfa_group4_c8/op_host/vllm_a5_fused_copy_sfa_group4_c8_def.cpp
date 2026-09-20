#include <vector>
#include "register/op_def_registry.h"

namespace ops {
class VllmA5FusedCopySfaGroup4C8 : public OpDef {
public:
    explicit VllmA5FusedCopySfaGroup4C8(const char *name) : OpDef(name)
    {
        const std::vector<ge::DataType> q = {ge::DT_BF16, ge::DT_FLOAT16};
        const std::vector<ge::DataType> bytes = {ge::DT_INT8, ge::DT_INT8};
        const std::vector<ge::DataType> ints = {ge::DT_INT32, ge::DT_INT32};
        const std::vector<ge::Format> nd = {ge::FORMAT_ND, ge::FORMAT_ND};
        this->Input("query").ParamType(REQUIRED).DataType(q).Format(nd);
        this->Input("hbm_kv_bytes").ParamType(REQUIRED).DataType(bytes).Format(nd);
        this->Input("dram_kv_bytes").ParamType(REQUIRED).DataType(bytes).Format(nd);
        for (const char *input : {"hbm_block_table", "dram_block_table", "topk_slots",
             "topk_index", "per_query_miss_counts", "actual_seq_lengths_query",
             "resident_seq_lengths", "copy_src_ids", "copy_dst_slots", "copy_counts",
             "attention_slots", "attention_block_table"}) {
            this->Input(input).ParamType(REQUIRED).DataType(ints).Format(nd);
        }
        this->Output("hbm_kv_bytes").ParamType(REQUIRED).DataType(bytes).Format(nd);
        this->Output("attention_out").ParamType(REQUIRED).DataType(q).Format(nd);
        this->Attr("scale_value").AttrType(REQUIRED).Float();
        this->AICore().AddConfig("ascend950");
    }
};
OP_ADD(VllmA5FusedCopySfaGroup4C8);
}
