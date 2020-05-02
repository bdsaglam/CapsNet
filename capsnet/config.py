MNIST = dict(
    input_shape=(1, 28, 28),
    cnn_out_channels=256,
    cnn_kernel_size=9,
    cnn_stride=1,
    pc_num_capsules=8,
    pc_out_channels=32,
    pc_kernel_size=9,
    pc_stride=2,
    obj_num_capsules=10,
    obj_out_channels=16
)

CIFAR = dict(
    input_shape=(3, 32, 32),
    cnn_out_channels=256,
    cnn_kernel_size=9,
    cnn_stride=1,
    pc_num_capsules=8,
    pc_out_channels=32,
    pc_kernel_size=9,
    pc_stride=2,
    obj_num_capsules=10,
    obj_out_channels=16
)
