def getModel(params, input_shape, num_classes):
    if params.model_name == 'CNN':
        from models.CNN import CNN
        model = CNN(in_channels=input_shape[1], height=input_shape[2], width=input_shape[3], num_classes=num_classes)
    elif params.model_name == 'MLP':
        from models.MLP import MLP
        model = MLP(input_shape=input_shape, num_classes=num_classes)
    elif params.model_name == 'LSTM':
        from models.LSTM import LSTM
        model = LSTM(input_shape=input_shape, hidden_size=params.hidden_size, num_classes=num_classes)
    elif params.model_name == 'GRU':
        from models.GRU import GRU
        model = GRU(input_shape=input_shape, hidden_size=params.hidden_size, num_classes=num_classes)
    else:
        raise ValueError(f"Model {params.model} not supported")
    return model