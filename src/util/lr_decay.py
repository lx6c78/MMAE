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

