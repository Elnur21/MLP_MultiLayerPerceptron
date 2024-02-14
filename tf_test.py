import tensorflow as tf
# tf.config.list_physical_devices()
gpus = tf.config.list_logical_devices(device_type='GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)