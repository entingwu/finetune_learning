"""
Qwen3 张量形状调试脚本
=====================

目的：把 batch_size / num_tokens / hidden_size / num_heads / head_dim 这几个维度
在真实前向传播中的位置，一步步打印出来。

用法：
    python 01_debug_shapes.py                  # 默认：2层、batch=2、随机权重
    python 01_debug_shapes.py --layers 28      # 跑完整 28 层
    python 01_debug_shapes.py --batch 1        # 改 batch_size 观察第0维变化
    python 01_debug_shapes.py --pdb            # 在注意力内部断点，手动 inspect

在 VSCode 里 debug：
    直接在本文件或 01_Qwen3.py 的 GroupedQueryAttention.forward 里打断点，
    然后 F5 运行本文件即可。推荐断点位置：
        01_Qwen3.py:224   queries.view(...)      ← 2048 拆成 16×128 的那一刻
        01_Qwen3.py:259   attn_scores = ...      ← [b,16,t,t] 注意力矩阵
        01_Qwen3.py:268   context.reshape(...)   ← 16×128 拼回 2048

实现说明：本脚本不复制模型代码，而是用 forward hook 挂在真实的 nn.Module 上，
所以 01_Qwen3.py 改了，这里打印的形状会自动跟着变。
"""

import argparse
import importlib.util
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# 01_Qwen3.py 的模块名以数字开头，不能用 import 语句，只能走 importlib
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    "qwen3_impl", Path(__file__).parent / "01_Qwen3.py"
)
qwen3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qwen3)


# ---------------------------------------------------------------------------
# 打印工具
# ---------------------------------------------------------------------------
def fmt(t):
    """把张量格式化成 shape + dtype"""
    if isinstance(t, torch.Tensor):
        return f"{list(t.shape)}  {str(t.dtype).replace('torch.', '')}"
    return str(type(t).__name__)


def line(label, tensor, note=""):
    note = f"   # {note}" if note else ""
    print(f"  {label:<34} {fmt(tensor):<28}{note}")


def rule(title):
    print(f"\n{'─' * 100}\n{title}\n{'─' * 100}")


# ---------------------------------------------------------------------------
# Hook 注册：只对第 layer_idx 层打印，避免 28 层刷屏
# ---------------------------------------------------------------------------
def register_hooks(model, cfg, watch_layer=0):
    handles = []
    H = cfg["hidden_size"]
    NH = cfg["num_attention_heads"]
    NKV = cfg["num_key_value_heads"]
    HD = cfg["head_dim"]
    D_OUT = NH * HD

    def hook(label, note_in=None, note_out=None, show_input=False):
        def fn(mod, inp, out):
            if show_input:
                line(f"{label}  <input>", inp[0], note_in or "")
            tensor = out[0] if isinstance(out, tuple) else out
            line(label, tensor, note_out or "")
        return fn

    def add(mod, *a, **kw):
        handles.append(mod.register_forward_hook(hook(*a, **kw)))

    # --- 输入层 ---
    add(model.tok_emb, "tok_emb(in_idx)",
        note_out=f"[b, t, hidden_size={H}]  ← 整数id 变成 {H} 维向量")

    blk = model.trf_blocks[watch_layer]
    att = blk.att

    # --- 注意力：升维 ---
    add(att.q_proj, "  att.q_proj(x)",
        note_out=f"[b, t, d_out={D_OUT}]  ← Linear({H}→{NH}*{HD}) 升维")
    add(att.k_proj, "  att.k_proj(x)",
        note_out=f"[b, t, {NKV * HD}]  ← 只有 {NKV} 个KV头 (GQA)")
    add(att.v_proj, "  att.v_proj(x)",
        note_out=f"[b, t, {NKV * HD}]")

    # --- 注意力：拆头（q_norm/k_norm 恰好作用在拆头之后，白嫖它们看形状）---
    if att.q_norm is not None:
        add(att.q_norm, "  queries 拆头后",
            note_out=f"[b, num_heads={NH}, t, head_dim={HD}]")
        add(att.k_norm, "  keys    拆头后",
            note_out=f"[b, num_kv_groups={NKV}, t, head_dim={HD}]")

    # --- 注意力：降维回主干 ---
    add(att.o_proj, "  att.o_proj",
        note_in=f"[b, t, d_out={D_OUT}]  ← 16个头 reshape 拼回一条",
        note_out=f"[b, t, hidden_size={H}]  ← 降维，才能做残差",
        show_input=True)

    # --- 注意力模块整体：顺便看 KV Cache ---
    def att_hook(mod, inp, out):
        _, (k, v) = out
        line("  → KV Cache 中的 keys", k, "decode 时这一维会不断变长")
    handles.append(att.register_forward_hook(att_hook))

    # --- FFN ---
    add(blk.ff.gate_proj, "  ff.gate_proj",
        note_out=f"[b, t, intermediate={cfg['intermediate_size']}]  ← 升维")
    add(blk.ff.down_proj, "  ff.down_proj",
        note_out=f"[b, t, hidden_size={H}]  ← 降回主干")

    # --- 输出层 ---
    add(model.final_norm, "final_norm", note_out=f"[b, t, hidden_size={H}]")
    add(model.out_head, "out_head",
        note_out=f"[b, t, vocab_size={cfg['vocab_size']}]  ← 每个token对全词表打分")

    return handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2,
                    help="Transformer 层数，默认 2（只看形状不需要 28 层）")
    ap.add_argument("--batch", type=int, default=2, help="batch_size")
    ap.add_argument("--pdb", action="store_true", help="在注意力内部断点")
    args = ap.parse_args()

    torch.manual_seed(0)

    # 形状调试用 float32 + CPU 最稳（MPS 对 bfloat16 的支持有坑）
    cfg = dict(qwen3.QWEN_CONFIG_06_B)
    cfg["num_hidden_layers"] = args.layers
    cfg["torch_dtype"] = torch.float32
    cfg["max_position_embeddings"] = 4096      # 省掉 40960 长度的 RoPE 表
    device = "cpu"

    # ---------------- 配置表 ----------------
    H, NH, NKV, HD = (cfg["hidden_size"], cfg["num_attention_heads"],
                      cfg["num_key_value_heads"], cfg["head_dim"])
    rule("配置：模型维度（架构固定，与输入无关）")
    print(f"  hidden_size          = {H:>6}   主干宽度，每个 token 的向量长度")
    print(f"  num_attention_heads  = {NH:>6}   Q 头数")
    print(f"  num_key_value_heads  = {NKV:>6}   KV 头数 (GQA)")
    print(f"  head_dim             = {HD:>6}   每个头的宽度")
    print(f"  d_out = {NH} * {HD}   = {NH * HD:>6}   注意力内部宽度")
    print(f"\n  ⚠  d_out({NH * HD}) != hidden_size({H})  →  所以需要 q_proj 升维 / o_proj 降维")

    tokenizer = qwen3.Qwen3Tokenizer(
        tokenizer_file_path="model/Qwen3-0.6B-Base/tokenizer.json")
    ids = tokenizer.encode("你好，今天天气真好啊")
    input_ids = torch.tensor(ids).unsqueeze(0).repeat(args.batch, 1).to(device)

    rule("输入：数据维度（由你喂进去的数据决定）")
    print(f"  batch_size  = {input_ids.shape[0]}   一次几条样本")
    print(f"  num_tokens  = {input_ids.shape[1]}   每条样本几个 token")
    line("input_ids", input_ids, "还只是整数 id，没有 hidden_size 这一维")
    print(f"  第0条的 token id: {ids}")
    print(f"  解码回来:        {[tokenizer.decode([i]) for i in ids]}")

    print("\n  建模中...（随机权重，只看形状不看内容）")
    model = qwen3.Qwen3Model(cfg).to(device).eval()

    if args.pdb:
        import pdb
        orig = qwen3.GroupedQueryAttention.forward

        def traced(self, x, mask, cos, sin, start_pos=0, cache=None):
            print("\n  >>> 进入 pdb。可以查看: x.shape / self.num_heads / "
                  "self.head_dim / self.d_out / mask.shape")
            pdb.set_trace()
            return orig(self, x, mask, cos, sin, start_pos, cache)
        qwen3.GroupedQueryAttention.forward = traced

    handles = register_hooks(model, cfg, watch_layer=0)

    # ---------------- Prefill ----------------
    rule(f"PREFILL 阶段（一次处理全部 {input_ids.shape[1]} 个 token）"
         f"   —— 下面只打印第 0 层，共 {args.layers} 层")
    kv_cache = {}
    with torch.no_grad():
        logits = model(input_ids, cache=kv_cache)

    print()
    line("最终 logits", logits, "[b, t, vocab_size]")
    k0, v0 = kv_cache[0]
    line("第0层 KV Cache 的 K", k0, f"[b, {NKV}, t, {HD}]")

    # 只在 prefill 打印，decode 换一套更精简的
    for h in handles:
        h.remove()

    # ---------------- Decode ----------------
    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    rule("DECODE 阶段（只喂 1 个新 token，num_tokens 变成 1）")
    line("next_input", next_id, "num_tokens = 1")

    handles = register_hooks(model, cfg, watch_layer=0)
    with torch.no_grad():
        logits2 = model(next_id, cache=kv_cache)
    for h in handles:
        h.remove()

    print()
    line("decode logits", logits2, "t 这一维只有 1")
    k1, _ = kv_cache[0]
    line("第0层 KV Cache 的 K", k1,
         f"seq 维从 {k0.shape[2]} 涨到 {k1.shape[2]}（缓存了历史）")

    # ---------------- 总结 ----------------
    rule("总结：四个维度分别管什么")
    print(f"""
  数据维度（运行时可变）              模型维度（架构固定）
  ────────────────────              ────────────────────
  batch_size = {input_ids.shape[0]}  ← 你的数据          hidden_size = {H}  ← 主干宽度，28层进出不变
  num_tokens = {input_ids.shape[1]}  ← 句子长度          head_dim    = {HD}  ← 只活在注意力内部

  一条完整数据流（b={input_ids.shape[0]}, t={input_ids.shape[1]}）：

      input_ids   [{input_ids.shape[0]}, {input_ids.shape[1]}]              整数 id
          │ tok_emb
      x           [{input_ids.shape[0]}, {input_ids.shape[1]}, {H}]        ← hidden_size 出现了
          │ q_proj  ({H} → {NH*HD})
      queries     [{input_ids.shape[0]}, {input_ids.shape[1]}, {NH*HD}]        ← d_out
          │ view + transpose（不做计算，只重新解释同一批数字）
      queries     [{input_ids.shape[0]}, {NH}, {input_ids.shape[1]}, {HD}]      ← num_heads × head_dim
          │ 每个头独立算注意力
      context     [{input_ids.shape[0]}, {NH}, {input_ids.shape[1]}, {HD}]
          │ transpose + reshape（16个头拼回一条）
      context     [{input_ids.shape[0]}, {input_ids.shape[1]}, {NH*HD}]        ← d_out
          │ o_proj  ({NH*HD} → {H})
      out         [{input_ids.shape[0]}, {input_ids.shape[1]}, {H}]        ← 回到 hidden_size，才能残差
          │ ×{args.layers} 层，形状始终不变
          │ out_head ({H} → {cfg['vocab_size']})
      logits      [{input_ids.shape[0]}, {input_ids.shape[1]}, {cfg['vocab_size']}]

  {NH} × {HD} = {NH*HD} = d_out  永远成立（定义）
  {NH} × {HD} = {NH*HD} ≠ {H} = hidden_size  本模型不相等（Qwen3 显式指定了 head_dim）
""")


if __name__ == "__main__":
    main()
