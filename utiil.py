import os
import torch


def save_model(_data_name, _model_name, _model):
    model_path = os.path.join('../edu_han/model_files/{}_{}.pth'.format(_data_name, _model_name))
    torch.save(_model.state_dict(), model_path)
    return


def load_model(_data_name, _model_name, _model):
    model_path = os.path.join('../edu_han/model_files/{}_{}.pth'.format(_data_name, _model_name))
    _model.load_state_dict(torch.load(model_path))

    return _model