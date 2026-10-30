"""Load a JiT ImageNet-pretrained checkpoint into a model whose label embedding
table has been resized.

The vanilla pretrained checkpoint is trained with num_classes=1000, so the
y_embedder.embedding_table has 1001 rows (0..999 ImageNet classes + row 1000
as the CFG-drop token). For RadImageNet "extended" fine-tuning we instantiate
the model with num_classes=1005 (5 region classes appended on top of
ImageNet), giving 1006 rows. Rows 0..1000 are copied from the pretrained
checkpoint; rows 1001..1005 stay at their fresh N(0, 0.02) init.

For the "plain" 5-class fine-tune (num_classes=5 → 6 rows) we simply skip
the label embedding from the pretrained ckpt — there is nothing meaningful
to transfer when the row count shrinks by 1000.

For multichannel fine-tunes (e.g. 3→4 or 3→5 channels for BraTS-multimodal)
the input PatchEmbed Conv2d and the output FinalLayer Linear get grown via
"keep-then-average": the first src_in_ch slots inherit the pretrained RGB
weights exactly, and the new slots (3..C-1) are initialised to the mean of
the pretrained RGB weights. No rescaling — symmetry-breaking between channels
happens immediately because slots 0..2 are distinct, and slot 3..C-1 start
as the channel-average prediction and differentiate via training.

Everything else (RoPE, blocks, final layer, time/patch embeds) loads with
strict=False; we print any unexpected/missing keys so silent mismatches are
visible.
"""
import torch


def _grow_input_conv_channels(src_w, target_in_ch):
    """Grow Conv2d weight [out, src_in, k, k] to [out, target_in, k, k].

    Keep the first src_in slots VERBATIM (preserve pretrained RGB filters),
    init the (target_in - src_in) new slots to the channel-average of the
    pretrained filters. No magnitude rescaling.
    """
    src_in = src_w.shape[1]
    if src_in == target_in_ch:
        return src_w
    avg = src_w.mean(dim=1, keepdim=True)                            # [out, 1, k, k]
    extras = avg.expand(-1, target_in_ch - src_in, -1, -1)           # [out, C-src, k, k]
    return torch.cat([src_w, extras], dim=1).clone()                 # [out, target_in, k, k]


def _grow_output_linear_channels(src_w, src_b, src_out_ch, target_out_ch, patch_sq):
    """Grow FinalLayer Linear weights/bias from src_out_ch -> target_out_ch image
    channels. The Linear maps hidden -> patch²·C with shape [patch²·C, hidden].

    Channel layout within the patch²·C flat output: model_jit.unpatchify reshapes
    via `(N, T, P²·C) -> (N, h, w, P, P, C)`, so the C dimension varies FASTEST
    (innermost) within each spatial pixel. Linear index i decodes to
    (p_h, p_w, c) with c = i % C — i.e., the rows are tiled as
    (pixel 0, channels 0..C-1), (pixel 1, channels 0..C-1), ...

    So the right reshape is [patch_sq, C, hidden]: spatial pixel OUTER, channel
    INNER. Then: first src_out_ch slots keep their pretrained projection
    VERBATIM; new slots (src_out_ch..target_out_ch-1) init to the channel
    average. No magnitude rescaling.
    """
    if src_out_ch == target_out_ch:
        return src_w, src_b
    hidden = src_w.shape[1]
    # [P²·C, hidden] -> [P², C, hidden]  (C is inner / fastest-varying)
    w_c = src_w.view(patch_sq, src_out_ch, hidden)                       # [P², src_C, hidden]
    w_avg = w_c.mean(dim=1, keepdim=True)                                # [P², 1, hidden]
    w_extras = w_avg.expand(-1, target_out_ch - src_out_ch, -1)          # [P², new_C, hidden]
    w_grown = torch.cat([w_c, w_extras], dim=1).clone()                  # [P², target_C, hidden]
    w_grown = w_grown.reshape(patch_sq * target_out_ch, hidden)          # [P²·target_C, hidden]

    b_c = src_b.view(patch_sq, src_out_ch)                               # [P², src_C]
    b_avg = b_c.mean(dim=1, keepdim=True)                                # [P², 1]
    b_extras = b_avg.expand(-1, target_out_ch - src_out_ch)              # [P², new_C]
    b_grown = torch.cat([b_c, b_extras], dim=1).clone()                  # [P², target_C]
    b_grown = b_grown.reshape(patch_sq * target_out_ch)                  # [P²·target_C]
    return w_grown, b_grown


def load_jit_pretrained(model_without_ddp, ckpt_path, args):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # Most JiT checkpoints store the trained weights under 'model'; the EMA
    # variants under 'model_ema1' / 'model_ema2'. The EMA tends to be the one
    # that scores best, so prefer it when present.
    if 'model_ema1' in ckpt:
        state = ckpt['model_ema1']
        print("Using model_ema1 weights from pretrained ckpt")
    elif 'model' in ckpt:
        state = ckpt['model']
        print("Using model weights from pretrained ckpt")
    else:
        state = ckpt
        print("Using raw checkpoint as state_dict")

    embed_key = 'net.y_embedder.embedding_table.weight'
    target_state = model_without_ddp.state_dict()

    if embed_key in state and embed_key in target_state:
        src = state[embed_key]
        tgt = target_state[embed_key]
        if src.shape == tgt.shape:
            pass  # exact match, normal load
        elif src.shape[1] == tgt.shape[1] and src.shape[0] <= tgt.shape[0]:
            # Grow: copy 0..src_rows-1 from the pretrained ckpt into a tensor
            # of target shape, keeping the freshly-initialised tail rows.
            grown = tgt.clone()
            grown[:src.shape[0]] = src
            state = dict(state)
            state[embed_key] = grown
            print(f"Grew label embedding {tuple(src.shape)} -> {tuple(tgt.shape)} "
                  f"(preserved first {src.shape[0]} rows, kept fresh init for "
                  f"rows {src.shape[0]}..{tgt.shape[0] - 1})")
        else:
            # Shrinking (num_classes=5 plain mode) or mismatched hidden size:
            # not safe to transfer. Drop the key so strict=False ignores it.
            state = {k: v for k, v in state.items() if k != embed_key}
            print(f"Dropping label embedding {tuple(src.shape)} -> {tuple(tgt.shape)} "
                  f"(incompatible shape; keeping fresh init)")

    # Channel-grow for multichannel fine-tunes (e.g. BraTS 3->4 or 3->5).
    # Only acts when the ckpt's channel count differs from the current model;
    # all existing 3-channel runs hit the no-op path and are unaffected.
    in_conv_key = 'net.x_embedder.proj1.weight'
    if in_conv_key in state and in_conv_key in target_state:
        src_w = state[in_conv_key]
        tgt_w = target_state[in_conv_key]
        if src_w.shape != tgt_w.shape and src_w.shape[0] == tgt_w.shape[0] \
                and src_w.shape[2:] == tgt_w.shape[2:]:
            state = dict(state)
            state[in_conv_key] = _grow_input_conv_channels(src_w, tgt_w.shape[1])
            print(f"Grew input PatchEmbed conv {tuple(src_w.shape)} -> {tuple(tgt_w.shape)} "
                  f"(first {src_w.shape[1]} ch verbatim, new {tgt_w.shape[1] - src_w.shape[1]} ch = mean of pretrained)")

    out_w_key = 'net.final_layer.linear.weight'
    out_b_key = 'net.final_layer.linear.bias'
    if out_w_key in state and out_w_key in target_state:
        src_w = state[out_w_key]
        tgt_w = target_state[out_w_key]
        if src_w.shape != tgt_w.shape and src_w.shape[1] == tgt_w.shape[1]:
            patch_sq = (model_without_ddp.net.patch_size ** 2)
            src_ch = src_w.shape[0] // patch_sq
            tgt_ch = tgt_w.shape[0] // patch_sq
            src_b = state.get(out_b_key)
            tgt_b = target_state.get(out_b_key)
            if src_b is not None and tgt_b is not None and src_b.shape != tgt_b.shape:
                new_w, new_b = _grow_output_linear_channels(
                    src_w, src_b, src_ch, tgt_ch, patch_sq
                )
                state = dict(state)
                state[out_w_key] = new_w
                state[out_b_key] = new_b
                print(f"Grew output FinalLayer linear {tuple(src_w.shape)}/{tuple(src_b.shape)} "
                      f"-> {tuple(tgt_w.shape)}/{tuple(tgt_b.shape)} "
                      f"(first {src_ch} ch verbatim, new {tgt_ch - src_ch} ch = mean of pretrained)")

    missing, unexpected = model_without_ddp.load_state_dict(state, strict=False)
    if missing:
        print(f"  missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    return model_without_ddp
