"""ყველა ექსპერიმენტის კონფიგი — არქიტექტურის მიხედვით დაჯგუფებული."""

TINY = [
    dict(arch='tiny', lr=1e-3, wd=1e-4, batch_size=128, epochs=20, augment=True,  class_weights=True,  dropout=None, tag='baseline'),
    dict(arch='tiny', lr=1e-3, wd=1e-4, batch_size=128, epochs=40, augment=True,  class_weights=True,  dropout=None, tag='long'),
    dict(arch='tiny', lr=5e-3, wd=1e-4, batch_size=128, epochs=20, augment=True,  class_weights=True,  dropout=None, tag='highlr'),
    dict(arch='tiny', lr=1e-3, wd=1e-4, batch_size=128, epochs=20, augment=False, class_weights=False, dropout=None, tag='noaug'),
]
SIMPLE = [
    dict(arch='simple', lr=1e-3, wd=0.0,  batch_size=128, epochs=30, augment=False, class_weights=False, dropout=None, tag='overfit'),
    dict(arch='simple', lr=1e-3, wd=5e-4, batch_size=128, epochs=30, augment=False, class_weights=False, dropout=None, tag='wd'),
    dict(arch='simple', lr=1e-3, wd=5e-4, batch_size=128, epochs=30, augment=True,  class_weights=True,  dropout=None, tag='aug_cw'),
]
CNN_BN = [
    dict(arch='cnn_bn', lr=1e-3, wd=5e-4, batch_size=128, epochs=30, augment=True, class_weights=True, dropout=0.0, tag='bn_only'),
    dict(arch='cnn_bn', lr=1e-3, wd=5e-4, batch_size=128, epochs=30, augment=True, class_weights=True, dropout=0.3, tag='drop03'),
    dict(arch='cnn_bn', lr=1e-3, wd=5e-4, batch_size=128, epochs=30, augment=True, class_weights=True, dropout=0.5, tag='drop05'),
    dict(arch='cnn_bn', lr=1e-3, wd=5e-4, batch_size=128, epochs=45, augment=True, class_weights=True, dropout=0.4, tag='best_long'),
]
CNN_BN_HP = [
    dict(arch='cnn_bn', lr=5e-4, wd=5e-4, batch_size=128, epochs=30, augment=True, class_weights=True, dropout=0.4, tag='hp_lr5e4'),
    dict(arch='cnn_bn', lr=3e-3, wd=5e-4, batch_size=128, epochs=30, augment=True, class_weights=True, dropout=0.4, tag='hp_lr3e3'),
    dict(arch='cnn_bn', lr=1e-3, wd=1e-5, batch_size=128, epochs=30, augment=True, class_weights=True, dropout=0.4, tag='hp_wd1e5'),
    dict(arch='cnn_bn', lr=1e-3, wd=1e-3, batch_size=128, epochs=30, augment=True, class_weights=True, dropout=0.4, tag='hp_wd1e3'),
    dict(arch='cnn_bn', lr=1e-3, wd=5e-4, batch_size=64,  epochs=30, augment=True, class_weights=True, dropout=0.4, tag='hp_bs64'),
    dict(arch='cnn_bn', lr=1e-3, wd=5e-4, batch_size=256, epochs=30, augment=True, class_weights=True, dropout=0.4, tag='hp_bs256'),
    dict(arch='cnn_bn', lr=2e-3, wd=1e-4, batch_size=256, epochs=30, augment=True, class_weights=True, dropout=0.3, tag='hp_combo_big'),
    dict(arch='cnn_bn', lr=5e-4, wd=1e-3, batch_size=64,  epochs=30, augment=True, class_weights=True, dropout=0.5, tag='hp_combo_small'),
]
RESNET = [
    dict(arch='resnet', lr=5e-4, wd=5e-4, batch_size=128, epochs=35, augment=True, class_weights=True,  dropout=0.3, tag='r1'),
    dict(arch='resnet', lr=1e-3, wd=5e-4, batch_size=128, epochs=35, augment=True, class_weights=True,  dropout=0.3, tag='r2'),
    dict(arch='resnet', lr=5e-4, wd=1e-4, batch_size=128, epochs=40, augment=True, class_weights=False, dropout=0.2, tag='r3_nocw_long'),
    dict(arch='resnet', lr=5e-4, wd=5e-4, batch_size=128, epochs=50, augment=True, class_weights=False, dropout=0.3, tag='r4_best'),
]
