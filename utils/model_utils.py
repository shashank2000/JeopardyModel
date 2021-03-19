def freeze_everything_except_top(model, num_layers):
    params = [p for p in model.parameters()]
    total_layers = len(params)
    i = 0
    while i < total_layers - num_layers:
        params[i].requires_grad = False
        i += 1

