import cudamat as cm
import gpu_lock2 as gpu_lock
import h5py
import sys
import os
import numpy as np
# import matplotlib.pyplot as plt
# plt.ion()
from time import sleep
import pdb
import datetime
import time
import config_pb2
from google.protobuf import text_format

class Param(object):
  def __init__(self, w, config=None):
    if type(w) == np.ndarray:
      self.w_ = cm.CUDAMatrix(w)
    elif type(w) == tuple:
      self.w_ = cm.empty(w)
    else:
      self.w_ = w
    self.dw_ = cm.empty_like(self.w_)
    self.dw_history_ = cm.empty_like(self.w_)
    self.dw_history_.assign(0)
    self.dw_.assign(0)
    self.t_ = 0
    self.rms_prop_ = config.rms_prop
    self.rms_prop_factor_ = config.rms_prop_factor
    if self.rms_prop_:
      self.rms_prop_history_ = cm.empty_like(self.dw_)
      self.rms_prop_history_.assign(1)

    if config is None:
      pass
    elif config.init_type == config_pb2.Param.CONSTANT:
      self.w_.assign(config.scale)
    elif config.init_type == config_pb2.Param.GAUSSIAN:
      self.w_.fill_with_randn()
      self.w_.mult(config.scale)
    elif config.init_type == config_pb2.Param.UNIFORM:
      self.w_.fill_with_rand()
      self.w_.subtract(0.5)
      self.w_.mult(2 * config.scale)
    elif config.init_type == config_pb2.Param.LSTM_BIAS:
      init_bias = [config.input_gate_bias, config.forget_gate_bias, config.input_bias, config.output_gate_bias]
      self.w_.reshape((-1, 4))
      for i in xrange(4):
        self.w_.slice(i, (i+1)).assign(init_bias[i])
      self.w_.reshape((-1, 1))
    elif config.init_type == config_pb2.Param.LSTM_SPATIAL_BIAS:
      init_bias = [config.input_gate_bias, config.forget_gate_bias, config.forget_gate_bias, config.input_bias, config.output_gate_bias]
      self.w_.reshape((-1, 5))
      for i in xrange(5):
        self.w_.slice(i, (i+1)).assign(init_bias[i])
      self.w_.reshape((-1, 1))
    elif config.init_type == config_pb2.Param.PRETRAINED:
      f = h5py.File(config.file_name)
      mat = f[config.dataset_name].value
      if len(mat.shape) == 1:
        mat = mat.reshape(1, -1)
      assert self.w_.shape == mat.shape
      self.w_.overwrite(mat)
      f.close()
    else:
      raise Exception('Unknown parameter initialization.')

    self.eps_ = config.epsilon
    self.momentum_ = config.momentum
    self.l2_decay_ = config.l2_decay
    self.gradient_clip_ = config.gradient_clip
    self.eps_decay_factor = config.eps_decay_factor
    self.eps_decay_after = config.eps_decay_after

  def __repr__(self):
    return self.w_.asarray().__repr__()
  
  def __str__(self):
    return self.w_.asarray().__str__()

  def Load(self, f, name):
    if name in f.keys():
      self.w_.overwrite(f[name].value)
      self.dw_history_.overwrite(f['%s_grad' % name].value)
      if self.rms_prop_:
        self.rms_prop_history_.overwrite(f['%s_rms_prop' % name].value)
      self.t_ = f.attrs.get('%s_t' % name, 0)
    else:
      print "%s not found." % name

  def Save(self, f, name):
    w_dset = f.create_dataset(name, self.w_.shape, dtype=np.float32)
    w_dset[:, :] = self.w_.asarray()
    w_dset = f.create_dataset('%s_grad' % name, self.dw_history_.shape, dtype=np.float32)
    w_dset[:, :] = self.dw_history_.asarray()
    if self.rms_prop_:
      w_dset = f.create_dataset('%s_rms_prop' % name, self.rms_prop_history_.shape, dtype=np.float32)
      w_dset[:, :] = self.rms_prop_history_.asarray()
    f.attrs.__setitem__('%s_t' % name, self.t_)

  def GetW(self):
    return self.w_
  
  def GetdW(self):
    return self.dw_

  def Update(self):
    if self.eps_decay_after > 0:
      eps = self.eps_ * np.power(self.eps_decay_factor, self.t_ / self.eps_decay_after)
    else:
      eps = self.eps_
    self.dw_history_.mult(self.momentum_)
    if self.l2_decay_ > 0:
      self.dw_.add_mult(self.w_, mult=self.l2_decay_)
    if self.rms_prop_:
      self.rms_prop_history_.rms_prop(self.dw_, self.rms_prop_factor_)
      self.dw_.divide(self.rms_prop_history_)
    self.dw_history_.add_mult(self.dw_, -self.eps_)
    if self.gradient_clip_ > 0:
      self.dw_history_.upper_bound_mod(self.gradient_clip_)
    self.w_.add(self.dw_history_)
    self.t_ += 1

def ReadDataProto(fname):
  data_pb = config_pb2.Data()
  with open(fname, 'r') as pbtxt:
    text_format.Merge(pbtxt.read(), data_pb)
  return data_pb

def ReadModelProto(fname):
  data_pb = config_pb2.Model()
  with open(fname, 'r') as pbtxt:
    text_format.Merge(pbtxt.read(), data_pb)
  return data_pb

def WritePbtxt(proto, fname):
  with open(fname, 'w') as f:
    text_format.PrintMessage(proto, f)

def LockGPU(max_retries=10, board=-1):
  # retry_count = 0
  # while board == -1 and retry_count < max_retries:
  #   board = gpu_lock.obtain_lock_id()
  #   if board == -1:
  #     sleep(1)
  #     retry_count += 1
  # if board == -1:
  #   print 'No GPU board available.'
  #   sys.exit(1)
  # else:
  #   cm.cuda_set_device(board)
  #   cm.cublas_init()
  board = 3
  cm.cuda_set_device(board)
  cm.cublas_init()
  return board

def FreeGPU(board):
  cm.cublas_shutdown()


