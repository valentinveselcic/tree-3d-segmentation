weight = 'exp/custom_trees/SpUNet-scratch-partial-revised/model/model_best.pth'
resume = False
evaluate = True
test_only = False
seed = 6130267
save_path = 'exp/custom_trees/SpUNet-scratch-partial-revised'
num_worker = 16
batch_size = 12
gradient_accumulation_steps = 1
batch_size_val = None
batch_size_test = None
epoch = 400
eval_epoch = 100
clip_grad = None
sync_bn = False
enable_amp = False
amp_dtype = 'float16'
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False
enable_wandb = True
wandb_project = 'pointcept'
wandb_key = None
mix_prob = 0.8
param_dicts = None
hooks = [
    dict(type='CheckpointLoader'),
    dict(type='ModelHook'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(
        type='SemSegEvaluator',
        write_cls_iou=True,
        eval_boundary=True,
        tau=0.02,
        theta=0.02,
        class_pair=(0, 1)),
    dict(type='CheckpointSaver', save_freq=None),
    dict(type='PreciseEvaluator', test_last=False)
]
train = dict(type='DefaultTrainer')
test = dict(type='SemSegTester', verbose=True)
model = dict(
    type='DefaultSegmentor',
    backbone=dict(
        type='SpUNet-v1m1',
        in_channels=6,
        num_classes=4,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2)),
    criteria=[dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1)])
optimizer = dict(
    type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0001, nesterov=True)
scheduler = dict(
    type='OneCycleLR',
    max_lr=0.05,
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=10000.0)
dataset_type = 'DefaultDataset'
data_root = 'data/Pregledan dataset/data'
data = dict(
    num_classes=4,
    ignore_index=-1,
    names=['trunk', 'branch', 'wire', 'grass'],
    train=dict(
        type='DefaultDataset',
        split='train',
        data_root='data/Pregledan dataset/data',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(
                type='RandomDropout',
                dropout_ratio=0.2,
                dropout_application_ratio=0.2),
            dict(
                type='RandomRotate',
                angle=[-1, 1],
                axis='z',
                center=[0, 0, 0],
                p=0.5),
            dict(
                type='RandomRotate',
                angle=[-0.015625, 0.015625],
                axis='x',
                p=0.5),
            dict(
                type='RandomRotate',
                angle=[-0.015625, 0.015625],
                axis='y',
                p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.5),
            dict(type='RandomJitter', sigma=0.005, clip=0.02),
            dict(
                type='ElasticDistortion',
                distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type='ChromaticAutoContrast', p=0.2, blend_factor=None),
            dict(type='ChromaticTranslation', p=0.95, ratio=0.05),
            dict(type='ChromaticJitter', p=0.95, std=0.05),
            dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='SphereCrop', point_max=100000, mode='random'),
            dict(type='CenterShift', apply_z=False),
            dict(type='NormalizeColor'),
            dict(type='ShufflePoint'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment'),
                feat_keys=['coord', 'color'])
        ],
        test_mode=False,
        loop=4),
    val=dict(
        type='DefaultDataset',
        split='val',
        data_root='data/Pregledan dataset/data',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(type='Copy', keys_dict=dict(segment='origin_segment')),
            dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True,
                return_inverse=True),
            dict(type='CenterShift', apply_z=False),
            dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'origin_segment',
                      'inverse'),
                feat_keys=['coord', 'color'])
        ],
        test_mode=False),
    test=dict(
        type='DefaultDataset',
        split='val',
        data_root='data/Pregledan dataset/data',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(type='NormalizeColor')
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='test',
                return_grid_coord=True),
            crop=None,
            post_transform=[
                dict(type='CenterShift', apply_z=False),
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'grid_coord', 'index'),
                    feat_keys=['coord', 'color'])
            ],
            aug_transform=[[{
                'type': 'RandomRotateTargetAngle',
                'angle': [0],
                'axis': 'z',
                'center': [0, 0, 0],
                'p': 1
            }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [0.5],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [1],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [1.5],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [0],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }, {
                               'type': 'RandomScale',
                               'scale': [0.95, 0.95]
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [0.5],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }, {
                               'type': 'RandomScale',
                               'scale': [0.95, 0.95]
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [1],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }, {
                               'type': 'RandomScale',
                               'scale': [0.95, 0.95]
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [1.5],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }, {
                               'type': 'RandomScale',
                               'scale': [0.95, 0.95]
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [0],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }, {
                               'type': 'RandomScale',
                               'scale': [1.05, 1.05]
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [0.5],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }, {
                               'type': 'RandomScale',
                               'scale': [1.05, 1.05]
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [1],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }, {
                               'type': 'RandomScale',
                               'scale': [1.05, 1.05]
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [1.5],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }, {
                               'type': 'RandomScale',
                               'scale': [1.05, 1.05]
                           }], [{
                               'type': 'RandomFlip',
                               'p': 1
                           }]])))
