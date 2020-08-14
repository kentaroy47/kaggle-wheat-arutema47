import gc
from effdet import get_efficientdet_config, EfficientDet, DetBenchEval
def load_net(checkpoint_path):
    config = get_efficientdet_config('tf_efficientdet_d5')
    devnet = EfficientDet(config, pretrained_backbone=False)

    config.num_classes = 1
    config.image_size=train_size
    devnet.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

    checkpoint = torch.load(checkpoint_path)
    devnet.load_state_dict(checkpoint['model_state_dict'])

    del checkpoint
    gc.collect()

    devnet = DetBenchEval(devnet, config)
    devnet.eval();
    return devnet.cuda()

