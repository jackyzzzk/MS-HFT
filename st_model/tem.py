from st_model.models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, MCformer
import torch
import torch.nn as nn

class tem_model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.model = self._build_model().to(self.device)
    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'MCformer': MCformer
        }
        model = model_dict[self.args.tem_model].Model(self.args).float()

        # if self.args.use_gpu:
        #     model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    def get_item(self, x, y):
        x_mark = torch.zeros((x.shape[0], x.shape[1], 1))
        y_mark = torch.zeros((y.shape[0], y.shape[1], 1))
        return x_mark, y_mark
    def forward(self, x, label):
        label = label.unsqueeze(-1)
        x_mark, y_mark = self.get_item(x, label)
        output = self.model(x, x_mark, label, y_mark)
        return output