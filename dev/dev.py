from importlib import reload

import deep_morpho.save_results_template.load_args as la

reload(la)

yaml_str = """
activation_P: 4
alpha_init: 0
args_thresh_penalization:
  coef: 0.005
  degree: 4
  detach_weights: true
atomic_element: bise
batch_size: 256
constant_activation_P: true
constant_weight_P: true
dataset_path: generate
dataset_type: diskorect
do_thresh_penalization: false
experiment_name: Test_save
experiment_subname: dilation
first_batch_pen: 1
freq_imgs: 500
in_ram: true
init_weight_identity: false
kernel_size: 7
learning_rate: 0.1
loss: !!python/object:general.nn.loss.dice_loss.DiceLoss
  _backward_hooks: !!python/object/apply:collections.OrderedDict
  - []
  _buffers: !!python/object/apply:collections.OrderedDict
  - []
  _forward_hooks: !!python/object/apply:collections.OrderedDict
  - []
  _forward_pre_hooks: !!python/object/apply:collections.OrderedDict
  - []
  _is_full_backward_hook: null
  _load_state_dict_pre_hooks: !!python/object/apply:collections.OrderedDict
  - []
  _modules: !!python/object/apply:collections.OrderedDict
  - []
  _non_persistent_buffers_set: !!set {}
  _parameters: !!python/object/apply:collections.OrderedDict
  - []
  _state_dict_hooks: !!python/object/apply:collections.OrderedDict
  - []
  eps: 1.0e-06
  reduction: mean
  training: true
morp_operation: !!python/object:general.array_morphology.SequentialMorpOperations
  _repr: SequentialMorpOperations(dilation(disk(3)))
  _selem_arg:
  - 3
  _selem_fn:
  - !!python/name:skimage.morphology.selem.disk ''
  _selems_original:
  - !!python/tuple
    - disk
    - 3
  device: cpu
  name: dilation
  operations:
  - dilation1
  - dilation2
  - erosion3
  - dilation4
  - erosion5
  return_numpy_array: false
  selems:
  - !!python/object/apply:numpy.core.multiarray._reconstruct
    args:
    - !!python/name:numpy.ndarray ''
    - !!python/tuple
      - 0
    - !!binary |
      Yg==
    state: !!python/tuple
    - 1
    - !!python/tuple
      - 7
      - 7
    - !!python/object/apply:numpy.dtype
      args:
      - u1
      - false
      - true
      state: !!python/tuple
      - 3
      - '|'
      - null
      - null
      - null
      - -1
      - -1
      - 0
    - false
    - !!binary |
      AAAAAQAAAAABAQEBAQAAAQEBAQEAAQEBAQEBAQABAQEBAQAAAQEBAQEAAAAAAQAAAA==
n_atoms: 1
n_epochs: 1
n_inputs: 10000
num_workers: 20
optimizer: !!python/name:torch.optim.adam.Adam ''
preprocessing: !!python/object:torchvision.transforms.transforms.ToTensor {}
random_gen_args:
  border: !!python/tuple
  - 4
  - 4
  max_shape: !!python/tuple
  - 20
  - 20
  max_shape_holes: !!python/tuple
  - 10
  - 10
  n_holes: 10
  n_shapes: 20
  noise_proba: 0.02
  p_invert: 0.5
  size: !!python/tuple
  - 50
  - 50
random_gen_fn: !!python/name:deep_morpho.datasets.generate_forms3.get_random_rotated_diskorect ''
share_weights: false
threshold_mode: sigmoid

"""

