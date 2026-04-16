def get_layer_id(n, num_layers):
    if 'mmae_core_block' in n:
        if 'levels' in n:
            try:
                level_idx = int(n.split('levels.')[1].split('.')[0])
                block_idx = int(n.split('blocks.')[1].split('.')[0])
                layer_id = level_idx * num_layers + block_idx
                if layer_id >= num_layers:
                    return num_layers - 1
                return layer_id
            except ValueError:
                return 0
        else:
            return 0
    else:
        return num_layers

def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    param_group_names = {}
    param_groups = {}
    num_layers = sum([len(layer.blocks) for layer in model.mmae_core_block.levels]) + 1 #

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())
