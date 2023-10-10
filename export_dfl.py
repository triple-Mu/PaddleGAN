import sys
import paddle
import paddle.nn as nn
import ppgan
import ppgan.models
import ppgan.models.builder
import ppgan.utils.filesystem
from ppgan.utils.config import get_config


class MyNet(nn.Layer):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.proj_weight = paddle.Tensor(model.proj_conv.weight.flatten().numpy())

    def forward(self, x):
        x = x / 255.
        x = self.model(x)
        x = paddle.clip(x, 0., 1.)
        x = x * 255.
        return x


def main():
    paddle.set_device('cpu')
    inputs_size = [1, 3, 1024, 1024]
    config_file = 'baidu/nafnet_n_dfl.yaml'
    weight_file = 'baidu/n-model-dfl/iter_96400_checkpoint.pdparams'
    export_to = 'baidu/n-model-dfl/model/model'
    cfg = get_config(config_file, overrides=None, show=True)
    model = ppgan.models.builder.build_model(cfg.model)
    model.setup_train_mode(is_train=False)
    generator = model.nets['generator']
    generator.eval()
    generator.export = True

    state_dicts = ppgan.utils.filesystem.load(weight_file)
    state_dicts = state_dicts['generator']
    generator.set_state_dict(state_dicts)
    generator = MyNet(generator)
    tmp = paddle.randn(inputs_size)
    generator(tmp)

    # param_mb = sum(m.numel() * m.element_size() for m in model.nets['generator'].parameters()) / (1 << 20)
    # print('Model size: {:.2f} MB'.format(float(param_mb)))

    input_spec = [paddle.static.InputSpec(shape=inputs_size, dtype='float32')]
    static_model = paddle.jit.to_static(generator, input_spec=input_spec)
    paddle.jit.save(static_model, export_to)


if __name__ == "__main__":
    main()
