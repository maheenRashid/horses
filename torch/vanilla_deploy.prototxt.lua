require 'nn'
local model = {}
-- warning: module ' [type HDF5Data]' not found
table.insert(model, {'Conv1', nn.SpatialConvolution(3, 16, 5, 5, 1, 1, 2, 2)})
-- warning: module 'ActivationTangH1 [type TanH]' not found
-- warning: module 'ActivationAbs1 [type AbsVal]' not found
table.insert(model, {'Pool1', nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()})
table.insert(model, {'Conv2', nn.SpatialConvolution(16, 48, 3, 3, 1, 1, 1, 1)})
-- warning: module 'ActivationTangH2 [type TanH]' not found
-- warning: module 'ActivationAbs2 [type AbsVal]' not found
table.insert(model, {'Pool2', nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()})
table.insert(model, {'Conv3', nn.SpatialConvolution(48, 64, 3, 3, 1, 1, 0, 0)})
-- warning: module 'ActivationTangH3 [type TanH]' not found
-- warning: module 'ActivationAbs3 [type AbsVal]' not found
table.insert(model, {'Pool3', nn.SpatialMaxPooling(3, 3, 2, 2, 0, 0):ceil()})
table.insert(model, {'Conv4', nn.SpatialConvolution(64, 64, 2, 2, 1, 1, 0, 0)})
-- warning: module 'ActivationTangH4 [type TanH]' not found
-- warning: module 'ActivationAbs4 [type AbsVal]' not found
table.insert(model, {'torch_view', nn.View(-1):setNumInputDims(3)})
table.insert(model, {'Dense1', nn.Linear(576, 100)})
-- warning: module 'ActivationTangH5 [type TanH]' not found
-- warning: module 'ActivationAbs5 [type AbsVal]' not found
table.insert(model, {'Dense2', nn.Linear(100, 10)})
-- warning: module 'loss [type Python]' not found
return model