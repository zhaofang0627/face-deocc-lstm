import sys
sys.path.append('/home/zhaofang/caffe/python/')
from util import *
import numpy as np

class LSTM(object):
  def __init__(self, lstm_config):
    self.name_ = lstm_config.name
    num_lstms = lstm_config.num_hid
    assert num_lstms  > 0
    self.num_lstms_   = num_lstms
    self.has_input_   = lstm_config.has_input
    self.has_embed_   = lstm_config.has_embed
    self.has_output_  = lstm_config.has_output
    self.has_atten_  = lstm_config.has_atten
    self.has_atten_hat_  = lstm_config.has_atten_hat
    self.has_recon_conv_  = lstm_config.has_recon_conv
    self.image_dims_ = lstm_config.image_dims
    self.image_side_ = lstm_config.image_side
    self.input_dims_  = lstm_config.input_dims
    self.input_patch_side_ = lstm_config.input_patch_side
    self.embed_dims_ = lstm_config.embed_dims
    self.output_patch_side_ = lstm_config.output_patch_side
    self.output_dims_ = lstm_config.output_dims
    self.use_relu_    = lstm_config.use_relu
    self.input_dropprob_  = lstm_config.input_dropprob
    self.output_dropprob_ = lstm_config.output_dropprob
    self.t_ = 0

    self.w_dense_  = Param((4 * num_lstms, num_lstms), lstm_config.w_dense)
    self.w_diag_   = Param((num_lstms, 3), lstm_config.w_diag)
    self.b_        = Param((4 * num_lstms, 1), lstm_config.b)
    self.param_list_ = [
      ('%s:w_dense' % self.name_, self.w_dense_),
      ('%s:w_diag'  % self.name_, self.w_diag_),
      ('%s:b'       % self.name_, self.b_),
    ]
    if self.has_input_:
      assert self.input_dims_ > 0
      self.w_input_ = Param((4 * num_lstms, self.input_dims_), lstm_config.w_input)
      self.param_list_.append(('%s:w_input' % self.name_, self.w_input_))
      if self.has_atten_:
        self.w_atten_ = Param((4, num_lstms), lstm_config.w_atten)
        self.param_list_.append(('%s:w_atten' % self.name_, self.w_atten_))
        self.b_atten_ = Param((4, 1), lstm_config.b_atten)
        self.param_list_.append(('%s:b_atten' % self.name_, self.b_atten_))
        # self.w_input_mr_32_ = Param((4 * num_lstms, self.input_dims_), lstm_config.w_input)
        # self.param_list_.append(('%s:w_input_mr_32' % self.name_, self.w_input_mr_32_))
      if self.has_embed_:
        self.w_embed_ = Param((self.input_dims_, self.embed_dims_), lstm_config.w_embed)
        self.param_list_.append(('%s:w_embed' % self.name_, self.w_embed_))
        self.b_embed_ = Param((self.input_dims_, 1), lstm_config.b_embed)
        self.param_list_.append(('%s:b_embed' % self.name_, self.b_embed_))
    if self.has_output_:
      assert self.output_dims_ > 0
      self.w_output_ = Param((self.output_dims_, num_lstms), lstm_config.w_output)
      self.param_list_.append(('%s:w_output' % self.name_, self.w_output_))
      self.b_output_ = Param((self.output_dims_, 1), lstm_config.b_output)
      self.param_list_.append(('%s:b_output' % self.name_, self.b_output_))
      if self.has_atten_hat_:
        self.w_atten_hat_ = Param((5, num_lstms), lstm_config.w_atten_hat)
        self.param_list_.append(('%s:w_atten_hat' % self.name_, self.w_atten_hat_))
        self.b_atten_hat_ = Param((5, 1), lstm_config.b_atten_hat)
        self.param_list_.append(('%s:b_atten_hat' % self.name_, self.b_atten_hat_))

  def HasInputs(self):
    return self.has_input_

  def HasOutputs(self):
    return self.has_output_

  def HasAttention(self):
    return self.has_atten_

  def HasAttentionHat(self):
    return self.has_atten_hat_

  def GetParams(self):
    return self.param_list_

  def SetBatchSize(self, batch_size, seq_length):
    assert batch_size > 0
    assert seq_length > 0
    self.batch_size_  = batch_size
    self.seq_length_  = seq_length
    self.gates_  = cm.empty((4 * self.num_lstms_, batch_size * seq_length))
    self.cell_   = cm.empty((self.num_lstms_, batch_size * seq_length))
    self.hidden_ = cm.empty((self.num_lstms_, batch_size * seq_length))
    self.gates_deriv_  = cm.empty_like(self.gates_)
    self.cell_deriv_   = cm.empty_like(self.cell_)
    self.hidden_deriv_ = cm.empty_like(self.hidden_)
    if self.has_input_:
      if self.has_atten_:
        self.para_affine_ = cm.empty((4, batch_size * seq_length))
        self.grid_x_ = cm.empty((self.input_patch_side_ * self.input_patch_side_, batch_size * seq_length))
        self.grid_y_ = cm.empty((self.input_patch_side_ * self.input_patch_side_, batch_size * seq_length))
        self.patches_ = cm.empty((self.embed_dims_, batch_size * seq_length))
        self.input_embed_ = cm.empty((self.input_dims_, batch_size * seq_length))
        self.grid_x_deriv_ = cm.empty_like(self.grid_x_)
        self.grid_y_deriv_ = cm.empty_like(self.grid_y_)
        self.para_affine_grid_deriv_ = cm.empty((4 * self.input_patch_side_ * self.input_patch_side_, batch_size * seq_length))
        self.para_affine_deriv_ = cm.empty_like(self.para_affine_)
        self.patch_deriv_ = cm.empty_like(self.patches_)
        self.input_embed_deriv_ = cm.empty_like(self.input_embed_)
        self.input_grid_deriv_ = cm.empty((self.embed_dims_ * self.image_side_ * self.image_side_, batch_size))
      if self.has_embed_:
        self.input_embed_ = cm.empty((self.input_dims_, batch_size * seq_length))
        self.input_embed_deriv_ = cm.empty_like(self.input_embed_)
    if self.has_output_:
      self.dec_step_ = cm.empty((self.image_dims_, batch_size * seq_length))
      if self.has_atten_hat_:
        self.para_atten_hat_ = cm.empty((5, batch_size * seq_length))
        self.F_x_hat_ = cm.empty((self.output_patch_side_ * self.image_side_, batch_size * seq_length))
        self.F_y_hat_ = cm.empty((self.output_patch_side_ * self.image_side_, batch_size * seq_length))
        self.output_patch_ = cm.empty((self.output_dims_, batch_size * seq_length))
        self.canvas_ = cm.empty((self.image_dims_, batch_size * seq_length))
        self.canvas_left_ = cm.empty((self.image_side_ * self.output_patch_side_, batch_size * seq_length))
        self.canvas_left_temp_ = cm.empty((self.image_side_ * self.output_patch_side_, batch_size))
        self.para_hat_deriv_ = cm.empty_like(self.para_atten_hat_)
        self.para_hat_deriv_row_ = cm.empty((1, batch_size))
        self.output_patch_deriv_ = cm.empty_like(self.output_patch_)
        self.para_hat_xy_deriv_ = cm.empty((6 * self.output_patch_side_ * self.image_side_, batch_size))
        self.para_hat_xy_deriv_row_ = cm.empty((self.output_patch_side_ * self.image_side_, batch_size))
        self.para_canv_hat_deriv_0 = cm.empty((self.image_side_ * self.image_side_, batch_size))
        self.para_canv_hat_deriv_1 = cm.empty((self.image_side_ * self.image_side_, batch_size))

    """
    if self.has_output_ and self.output_dropprob_ > 0:
      self.output_drop_mask_ = cm.empty_like(self.hiddenbatch_size, self.num_lstms_)) for i in xrange(seq_length)]
      self.output_intermediate_state_ = [cm.empty((batch_size, self.num_lstms_)) for i in xrange(seq_length)]
      self.output_intermediate_deriv_ = [cm.empty((batch_size, self.num_lstms_)) for i in xrange(seq_length)]

    if self.has_input_ and self.input_dropprob_ > 0:
      self.input_drop_mask_ = [cm.empty((batch_size, self.input_dims_)) for i in xrange(seq_length)]
      self.input_intermediate_state_ = [cm.empty((batch_size, self.input_dims_)) for i in xrange(seq_length)]
      self.input_intermediate_deriv_ = [cm.empty((batch_size, self.input_dims_)) for i in xrange(seq_length)]
    """

  def Load(self, f):
    for name, p in self.param_list_:
      p.Load(f, name)

  def Save(self, f):
    for name, p in self.param_list_:
      p.Save(f, name)

  def Fprop(self, input_frame=None, init_cell=None, init_hidden=None, occ_pre=None, input_mr=None, output_frame=None, prev_model_hidden=None, train=False):
    t = self.t_
    batch_size = self.batch_size_
    assert t >= 0
    assert t < self.seq_length_
    num_lstms = self.num_lstms_
    start = t * batch_size
    end = start + batch_size
    gates        = self.gates_.slice(start, end)
    cell_state   = self.cell_.slice(start, end)
    hidden_state = self.hidden_.slice(start, end)
    if t == 0:
      if init_cell is None:
        if input_frame is not None:
          assert self.has_input_
          if self.has_atten_:
            input_embed = self.input_embed_.slice(start, end)
            # average pooling
            cm.sum_para_derive(input_embed, input_frame, self.image_side_)
            input_embed.divide(self.image_side_*self.image_side_)
            gates.add_dot(self.w_input_.GetW(), input_embed)
          else:
            gates.add_dot(self.w_input_.GetW(), input_frame)
        gates.add_col_vec(self.b_.GetW())
        cm.lstm_fprop2_init(gates, cell_state, hidden_state, self.w_diag_.GetW())
      else:
        cell_state.add(init_cell)
        assert init_hidden is not None
        hidden_state.add(init_hidden)
    else:
      prev_start = start - batch_size
      prev_hidden_state = self.hidden_.slice(prev_start, start)
      prev_cell_state = self.cell_.slice(prev_start, start)
      if input_frame is not None: 
        assert self.has_input_
        if self.has_atten_:
          para_affine = self.para_affine_.slice(start, end)
          grid_x = self.grid_x_.slice(start, end)
          grid_y = self.grid_y_.slice(start, end)
          patches = self.patches_.slice(start, end)
          input_embed = self.input_embed_.slice(start, end)
          para_affine.add_dot(self.w_atten_.GetW(), prev_model_hidden)
          para_affine.add_col_vec(self.b_atten_.GetW())
          cm.get_atten_grid(grid_x, grid_y, para_affine, self.input_patch_side_, self.image_side_)
          cm.apply_sampling_grid_feat(patches, grid_x, grid_y, input_frame, self.input_patch_side_, self.image_side_)
          if self.has_embed_:
            input_embed.add_dot(self.w_embed_.GetW(), patches)
            input_embed.add_col_vec(self.b_embed_.GetW())
          else:
            # average pooling
            cm.sum_para_derive(input_embed, patches, self.input_patch_side_)
            input_embed.divide(self.input_patch_side_*self.input_patch_side_)
          gates.add_dot(self.w_input_.GetW(), input_embed)
        elif self.has_embed_:
          input_embed = self.input_embed_.slice(start, end)
          input_embed.add_dot(self.w_embed_.GetW(), input_frame)
          input_embed.add_col_vec(self.b_embed_.GetW())
          gates.add_dot(self.w_input_.GetW(), input_embed)
        else:
          gates.add_dot(self.w_input_.GetW(), input_frame)
      gates.add_dot(self.w_dense_.GetW(), prev_hidden_state)
      gates.add_col_vec(self.b_.GetW())
      cm.lstm_fprop2(gates, prev_cell_state, cell_state, hidden_state, self.w_diag_.GetW())

    if self.has_output_:
      assert output_frame is not None
      if self.has_atten_hat_:
        para_atten_hat = self.para_atten_hat_.slice(start, end)
        F_x_hat = self.F_x_hat_.slice(start, end)
        F_y_hat = self.F_y_hat_.slice(start, end)
        canvas_left = self.canvas_left_.slice(start, end)
        canvas = self.canvas_.slice(start, end)
        output_patch = self.output_patch_.slice(start, end)
        output_patch.add_dot(self.w_output_.GetW(), hidden_state)
        output_patch.add_col_vec(self.b_output_.GetW())
        para_atten_hat.add_dot(self.w_atten_hat_.GetW(), hidden_state)
        para_atten_hat.add_col_vec(self.b_atten_hat_.GetW())
        cm.get_glimpses_matrix(F_x_hat, F_y_hat, para_atten_hat, self.output_patch_side_, self.image_side_)
        cm.normalize_glmatrix(F_x_hat, F_y_hat, self.output_patch_side_, self.image_side_)
        cm.apply_glmatrix_hat(canvas_left, F_y_hat, para_atten_hat, output_patch, self.output_patch_side_, self.image_side_)
        cm.apply_glmatrix_hat(canvas, F_x_hat, para_atten_hat, canvas_left, self.output_patch_side_, self.image_side_)
        output_frame.add(canvas)
      else:
        output_frame.add_dot(self.w_output_.GetW(), hidden_state)
        output_frame.add_col_vec(self.b_output_.GetW())

    self.t_ += 1

  def BpropAndOutp(self, input_frame=None, input_deriv=None,
                   init_cell=None, init_hidden=None,
                   init_cell_deriv=None, init_hidden_deriv=None,
                   prev_model_hidden=None, prev_model_hidden_deriv=None,
                   input_mr=None, output_deriv=None):
    batch_size = self.batch_size_
    self.t_ -= 1

    t = self.t_
    assert t >= 0
    assert t < self.seq_length_
    num_lstms = self.num_lstms_
    start = t * batch_size
    end = start + batch_size
    gates        = self.gates_.slice(start, end)
    gates_deriv  = self.gates_deriv_.slice(start, end)
    cell_state   = self.cell_.slice(start, end)
    cell_deriv   = self.cell_deriv_.slice(start, end)
    hidden_state = self.hidden_.slice(start, end)
    hidden_deriv = self.hidden_deriv_.slice(start, end)

    if self.has_output_:
      assert output_deriv is not None  # If this lstm's output was used, it must get a deriv back.
      if self.has_atten_hat_:
        para_atten_hat = self.para_atten_hat_.slice(start, end)
        para_hat_deriv = self.para_hat_deriv_.slice(start, end)
        F_x_hat = self.F_x_hat_.slice(start, end)
        F_y_hat = self.F_y_hat_.slice(start, end)
        output_patch = self.output_patch_.slice(start, end)
        output_patch_deriv = self.output_patch_deriv_.slice(start, end)
        canvas_left  = self.canvas_left_.slice(start, end)
        canvas       = self.canvas_.slice(start, end)
        cm.get_para0123_derive(self.para_hat_xy_deriv_, F_x_hat, F_y_hat, para_atten_hat, self.output_patch_side_, self.image_side_)

        self.para_hat_xy_deriv_.get_row_slice(0, self.output_patch_side_ * self.image_side_, self.para_hat_xy_deriv_row_)
        cm.apply_glmatrix_hat(self.para_canv_hat_deriv_0, self.para_hat_xy_deriv_row_, para_atten_hat, canvas_left, self.output_patch_side_, self.image_side_)
        self.para_canv_hat_deriv_0.divide(self.output_patch_side_).mult(output_deriv).sum(0, self.para_hat_deriv_row_).divide(self.image_side_ * self.image_side_)
        para_hat_deriv.set_row_slice(0, 1, self.para_hat_deriv_row_)

        self.para_hat_xy_deriv_.get_row_slice(self.output_patch_side_ * self.image_side_, 2 * self.output_patch_side_ * self.image_side_, self.para_hat_xy_deriv_row_)
        cm.apply_glmatrix_hat(self.canvas_left_temp_, self.para_hat_xy_deriv_row_, para_atten_hat, output_patch, self.output_patch_side_, self.image_side_)
        cm.apply_glmatrix_hat(self.para_canv_hat_deriv_1, F_x_hat, para_atten_hat, self.canvas_left_temp_, self.output_patch_side_, self.image_side_)
        self.para_canv_hat_deriv_1.divide(self.output_patch_side_).mult(output_deriv).sum(0, self.para_hat_deriv_row_).divide(self.image_side_ * self.image_side_)
        para_hat_deriv.set_row_slice(1, 2, self.para_hat_deriv_row_)

        self.para_hat_xy_deriv_.get_row_slice(2 * self.output_patch_side_ * self.image_side_, 3 * self.output_patch_side_ * self.image_side_, self.para_hat_xy_deriv_row_)
        cm.apply_glmatrix_hat(self.para_canv_hat_deriv_0, self.para_hat_xy_deriv_row_, para_atten_hat, canvas_left, self.output_patch_side_, self.image_side_)
        self.para_hat_xy_deriv_.get_row_slice(3 * self.output_patch_side_ * self.image_side_, 4 * self.output_patch_side_ * self.image_side_, self.para_hat_xy_deriv_row_)
        cm.apply_glmatrix_hat(self.canvas_left_temp_, self.para_hat_xy_deriv_row_, para_atten_hat, output_patch, self.output_patch_side_, self.image_side_)
        cm.apply_glmatrix_hat(self.para_canv_hat_deriv_1, F_x_hat, para_atten_hat, self.canvas_left_temp_, self.output_patch_side_, self.image_side_)
        self.para_canv_hat_deriv_0.add(self.para_canv_hat_deriv_1).divide(self.output_patch_side_).mult(output_deriv).sum(0, self.para_hat_deriv_row_).divide(self.image_side_ * self.image_side_)
        para_hat_deriv.set_row_slice(2, 3, self.para_hat_deriv_row_)

        self.para_hat_xy_deriv_.get_row_slice(4 * self.output_patch_side_ * self.image_side_, 5 * self.output_patch_side_ * self.image_side_, self.para_hat_xy_deriv_row_)
        cm.apply_glmatrix_hat(self.para_canv_hat_deriv_0, self.para_hat_xy_deriv_row_, para_atten_hat, canvas_left, self.output_patch_side_, self.image_side_)
        self.para_hat_xy_deriv_.get_row_slice(5 * self.output_patch_side_ * self.image_side_, 6 * self.output_patch_side_ * self.image_side_, self.para_hat_xy_deriv_row_)
        cm.apply_glmatrix_hat(self.canvas_left_temp_, self.para_hat_xy_deriv_row_, para_atten_hat, output_patch, self.output_patch_side_, self.image_side_)
        cm.apply_glmatrix_hat(self.para_canv_hat_deriv_1, F_x_hat, para_atten_hat, self.canvas_left_temp_, self.output_patch_side_, self.image_side_)
        self.para_canv_hat_deriv_0.add(self.para_canv_hat_deriv_1).divide(self.output_patch_side_).mult(output_deriv).sum(0, self.para_hat_deriv_row_).divide(self.image_side_ * self.image_side_)
        para_hat_deriv.set_row_slice(3, 4, self.para_hat_deriv_row_)

        canvas.mult(-1, self.para_canv_hat_deriv_0)
        self.para_canv_hat_deriv_0.mult(output_deriv).sum(0, self.para_hat_deriv_row_).divide(self.image_side_ * self.image_side_)
        para_hat_deriv.set_row_slice(4, 5, self.para_hat_deriv_row_)

        self.w_atten_hat_.GetdW().add_dot(para_hat_deriv, hidden_state.T)
        self.b_atten_hat_.GetdW().add_sums(para_hat_deriv, axis=1)

        cm.get_output_patch_derive(output_patch_deriv, output_deriv, F_x_hat, F_y_hat, para_atten_hat, self.output_patch_side_, self.image_side_)

        self.w_output_.GetdW().add_dot(output_patch_deriv, hidden_state.T)
        self.b_output_.GetdW().add_sums(output_patch_deriv, axis=1)

        hidden_deriv.add_dot(self.w_atten_hat_.GetW().T, para_hat_deriv)
        hidden_deriv.add_dot(self.w_output_.GetW().T, output_patch_deriv)
      else:
        self.w_output_.GetdW().add_dot(output_deriv, hidden_state.T)
        self.b_output_.GetdW().add_sums(output_deriv, axis=1)
        hidden_deriv.add_dot(self.w_output_.GetW().T, output_deriv)

    if t == 0:
      if init_cell is None:
        assert self.has_input_
        cm.lstm_bprop2_init(gates, gates_deriv, cell_state, cell_deriv, hidden_deriv, self.w_diag_.GetW())
        cm.lstm_outp2_init(gates_deriv, cell_state, self.w_diag_.GetdW())
        self.b_.GetdW().add_sums(gates_deriv, axis=1)
        if self.has_atten_:
          input_embed = self.input_embed_.slice(start, end)
          self.w_input_.GetdW().add_dot(gates_deriv, input_embed.T)
          if input_deriv is not None:
            input_embed_deriv = self.input_embed_deriv_.slice(start, end)
            input_embed_deriv.add_dot(self.w_input_.GetW().T, gates_deriv)
            cm.dis_para_derive(input_deriv, input_embed_deriv, self.image_side_)
        else:
          self.w_input_.GetdW().add_dot(gates_deriv, input_frame.T)
          if input_deriv is not None:
            input_deriv.add_dot(self.w_input_.GetW().T, gates_deriv)
      else:
        init_hidden_deriv.add(hidden_deriv)
        init_cell_deriv.add(cell_deriv)
    else:
      prev_start = start - batch_size
      prev_hidden_state = self.hidden_.slice(prev_start, start)
      prev_hidden_deriv = self.hidden_deriv_.slice(prev_start, start)
      prev_cell_state   = self.cell_.slice(prev_start, start)
      prev_cell_deriv   = self.cell_deriv_.slice(prev_start, start)
      cm.lstm_bprop2(gates, gates_deriv, prev_cell_state, prev_cell_deriv,
                     cell_state, cell_deriv, hidden_deriv, self.w_diag_.GetW())
      cm.lstm_outp2(gates_deriv, prev_cell_state, cell_state, self.w_diag_.GetdW())
      self.b_.GetdW().add_sums(gates_deriv, axis=1)
      self.w_dense_.GetdW().add_dot(gates_deriv, prev_hidden_state.T)
      prev_hidden_deriv.add_dot(self.w_dense_.GetW().T, gates_deriv)
      if input_frame is not None:
        assert self.has_input_
        if self.has_atten_:
          para_affine_grid_deriv = self.para_affine_grid_deriv_.slice(start, end)
          para_affine_deriv = self.para_affine_deriv_.slice(start, end)
          grid_x = self.grid_x_.slice(start, end)
          grid_y = self.grid_y_.slice(start, end)
          patches = self.patches_.slice(start, end)
          grid_x_deriv = self.grid_x_deriv_.slice(start, end)
          grid_y_deriv = self.grid_y_deriv_.slice(start, end)
          patch_deriv = self.patch_deriv_.slice(start, end)
          input_embed = self.input_embed_.slice(start, end)
          input_embed_deriv = self.input_embed_deriv_.slice(start, end)
          self.w_input_.GetdW().add_dot(gates_deriv, input_embed.T)
          input_embed_deriv.add_dot(self.w_input_.GetW().T, gates_deriv)
          if self.has_embed_:
            self.w_embed_.GetdW().add_dot(input_embed_deriv, patches.T)
            self.b_embed_.GetdW().add_sums(input_embed_deriv, axis=1)
            patch_deriv.add_dot(self.w_embed_.GetW().T, input_embed_deriv)
          else:
            cm.dis_para_derive(patch_deriv, input_embed_deriv, self.input_patch_side_)
          if input_deriv is not None:
            self.input_grid_deriv_.assign(0)
            cm.get_grid_and_feat_derive(grid_x_deriv, grid_y_deriv, self.input_grid_deriv_, patch_deriv, grid_x, grid_y, input_frame, self.input_patch_side_, self.image_side_)
            cm.sum_para_derive(input_deriv, self.input_grid_deriv_, self.input_patch_side_)
          else:
            cm.get_grid_derive_feat(grid_x_deriv, grid_y_deriv, patch_deriv, grid_x, grid_y, input_frame, self.input_patch_side_, self.image_side_)
          cm.get_para_atten_derive(para_affine_grid_deriv, grid_x_deriv, grid_y_deriv, self.input_patch_side_, self.image_side_)
          cm.sum_para_derive(para_affine_deriv, para_affine_grid_deriv, self.input_patch_side_)

          self.w_atten_.GetdW().add_dot(para_affine_deriv, prev_model_hidden.T)
          self.b_atten_.GetdW().add_sums(para_affine_deriv, axis=1)
          prev_model_hidden_deriv.add_dot(self.w_atten_.GetW().T, para_affine_deriv)
        elif self.has_embed_:
          input_embed = self.input_embed_.slice(start, end)
          input_embed_deriv = self.input_embed_deriv_.slice(start, end)
          self.w_input_.GetdW().add_dot(gates_deriv, input_embed.T)
          input_embed_deriv.add_dot(self.w_input_.GetW().T, gates_deriv)
          self.w_embed_.GetdW().add_dot(input_embed_deriv, input_frame.T)
          self.b_embed_.GetdW().add_sums(input_embed_deriv, axis=1)
          if input_deriv is not None:
            input_deriv.add_dot(self.w_embed_.GetW().T, input_embed_deriv)
        else:
          self.w_input_.GetdW().add_dot(gates_deriv, input_frame.T)
          if input_deriv is not None:
            input_deriv.add_dot(self.w_input_.GetW().T, gates_deriv)

  def GetCurrentCellState(self):
    t = self.t_ - 1
    assert t >= 0 and t < self.seq_length_
    batch_size = self.batch_size_
    return self.cell_.slice(t * batch_size, (t+1) * batch_size)

  def GetCurrentCellDeriv(self):
    t = self.t_ - 1
    assert t >= 0 and t < self.seq_length_
    batch_size = self.batch_size_
    return self.cell_deriv_.slice(t * batch_size, (t+1) * batch_size)

  def GetCurrentHiddenState(self):
    t = self.t_ - 1
    assert t >= 0 and t < self.seq_length_
    batch_size = self.batch_size_
    return self.hidden_.slice(t * batch_size, (t+1) * batch_size)
  
  def GetCurrentHiddenDeriv(self):
    t = self.t_ - 1
    assert t >= 0 and t < self.seq_length_
    batch_size = self.batch_size_
    return self.hidden_deriv_.slice(t * batch_size, (t+1) * batch_size)

  def Update(self):
    self.w_dense_.Update()
    self.w_diag_.Update()
    self.b_.Update()
    if self.has_input_:
      self.w_input_.Update()
      if self.has_atten_:
        self.w_atten_.Update()
        self.b_atten_.Update()
        # self.w_input_mr_32_.Update()
      if self.has_embed_:
        self.w_embed_.Update()
        self.b_embed_.Update()
    if self.has_output_:
      self.w_output_.Update()
      self.b_output_.Update()
      if self.has_atten_hat_:
        self.w_atten_hat_.Update()
        self.b_atten_hat_.Update()

  def Display(self, fig=1):
    plt.figure(2*fig)
    plt.clf()
    name = ['h', 'c', 'i', 'f', 'a', 'o']
    for i in xrange(self.seq_length_):
      state = self.state_[i].asarray()
      for j in xrange(6):
        plt.subplot(3 * self.seq_length_, 6, 18*i+j+1)
        start = j * self.num_lstms_
        end = (j+1) * self.num_lstms_
        plt.imshow(state[:, start:end])
        _, labels = plt.xticks()
        plt.gca().xaxis.set_visible(False)
        plt.gca().yaxis.set_visible(False)
        #plt.setp(labels, rotation=45)
        
        plt.subplot(3 * self.seq_length_, 6, 18*i+j+7)
        plt.hist(state[:, start:end].flatten(), 100)
        _, labels = plt.xticks()
        plt.gca().yaxis.set_visible(False)
        plt.setp(labels, rotation=45)
        
        plt.subplot(3 * self.seq_length_, 6, 18*i+j+13)
        plt.hist(state[:, start:end].mean(axis=0).flatten(), 100)
        _, labels = plt.xticks()
        plt.gca().yaxis.set_visible(False)
        plt.setp(labels, rotation=45)
        plt.title('%s:%.3f' % (name[j],state[:, start:end].mean()))

    plt.draw()
    
    plt.figure(2*fig+1)
    plt.clf()
    name = ['w_dense', 'w_diag', 'b', 'w_input']
    ws = [self.w_dense_, self.w_diag_, self.b_, self.w_input_]
    l = len(ws)
    for i in xrange(l):
      w = ws[i]
      plt.subplot(1, l, i+1)
      plt.hist(w.GetW().asarray().flatten(), 100)
      _, labels = plt.xticks()
      plt.setp(labels, rotation=45)
      plt.title(name[i])
    plt.draw()

  def Reset(self):
    self.t_ = 0
    self.gates_.assign(0)
    self.gates_deriv_.assign(0)
    self.cell_.assign(0)
    self.cell_deriv_.assign(0)
    self.hidden_.assign(0)
    self.hidden_deriv_.assign(0)
    self.b_.GetdW().assign(0)
    self.w_dense_.GetdW().assign(0)
    self.w_diag_.GetdW().assign(0)
    if self.has_input_:
      self.w_input_.GetdW().assign(0)
      if self.has_atten_:
        self.para_affine_.assign(0)
        self.grid_x_.assign(0)
        self.grid_y_.assign(0)
        self.patches_.assign(0)
        self.input_embed_.assign(0)
        self.para_affine_grid_deriv_.assign(0)
        self.para_affine_deriv_.assign(0)
        self.patch_deriv_.assign(0)
        self.input_embed_deriv_.assign(0)
        self.grid_x_deriv_.assign(0)
        self.grid_y_deriv_.assign(0)
        self.w_atten_.GetdW().assign(0)
        self.b_atten_.GetdW().assign(0)
        # self.w_input_mr_32_.GetdW().assign(0)
      if self.has_embed_:
        self.input_embed_.assign(0)
        self.input_embed_deriv_.assign(0)
        self.w_embed_.GetdW().assign(0)
        self.b_embed_.GetdW().assign(0)
    if self.has_output_:
      self.w_output_.GetdW().assign(0)
      self.b_output_.GetdW().assign(0)
      self.dec_step_.assign(0)
      if self.has_atten_hat_:
        self.para_atten_hat_.assign(0)
        self.F_x_hat_.assign(0)
        self.F_y_hat_.assign(0)
        self.output_patch_.assign(0)
        self.canvas_.assign(0)
        self.canvas_left_.assign(0)
        self.canvas_left_temp_.assign(0)
        self.para_hat_deriv_.assign(0)
        self.para_hat_deriv_row_.assign(0)
        self.output_patch_deriv_.assign(0)
        self.para_hat_xy_deriv_.assign(0)
        self.para_hat_xy_deriv_row_.assign(0)
        self.para_canv_hat_deriv_0.assign(0)
        self.para_canv_hat_deriv_1.assign(0)
        self.w_atten_hat_.GetdW().assign(0)
        self.b_atten_hat_.GetdW().assign(0)

  def GetInputDims(self):
    return self.input_dims_
  
  def GetOutputDims(self):
    return self.output_dims_

  def GetAllStates(self):
    return self.state_

class LSTMStack(object):
  def __init__(self):
    self.models_ = []
    self.num_models_ = 0

  def Add(self, model):
    self.models_.append(model)
    self.num_models_ += 1

  def Fprop(self, input_frame=None, init_cell=[], init_hidden=[], occ_pre=None, input_mr=None, output_frame=None, train=False):
    num_models = self.num_models_
    num_init_cell = len(init_cell)
    assert num_init_cell == 0 or num_init_cell == num_models
    assert len(init_hidden) == num_init_cell
    for m, model in enumerate(self.models_):
      this_input_frame  = input_frame if m == 0 else self.models_[m-1].GetCurrentHiddenState()
      this_init_cell    = init_cell[num_models-1-m] if num_init_cell > 0 else None
      this_init_hidden  = init_hidden[num_models-1-m] if num_init_cell > 0 else None
      this_occ_pre = occ_pre if m == num_models - 1 else None
      this_input_mr = input_mr if m == 0 else None
      this_output_frame = output_frame if m == num_models - 1 else None
      this_prev_model_hidden = self.models_[num_models - 1].GetCurrentHiddenState() if m == 0 and model.t_ > 0 else None
      model.Fprop(input_frame=this_input_frame,
                  init_cell=this_init_cell,
                  init_hidden=this_init_hidden,
                  occ_pre=this_occ_pre,
                  input_mr=this_input_mr,
                  output_frame=this_output_frame,
                  prev_model_hidden=this_prev_model_hidden,
                  train=train)

  def BpropAndOutp(self, input_frame=None, input_deriv=None,
                   init_cell=[], init_hidden=[], init_cell_deriv=[],
                   init_hidden_deriv=[], input_mr=None, output_deriv=None):
    num_models = self.num_models_
    num_init_cell = len(init_cell)
    assert num_init_cell == 0 or num_init_cell == num_models
    assert len(init_hidden) == num_init_cell
    assert len(init_cell_deriv) == num_init_cell
    assert len(init_hidden_deriv) == num_init_cell
    for m in xrange(num_models-1, -1, -1):
      model = self.models_[m]
      this_input_frame  = input_frame if m == 0 else self.models_[m-1].GetCurrentHiddenState()
      this_input_deriv  = input_deriv if m == 0 else self.models_[m-1].GetCurrentHiddenDeriv() 
      this_init_cell    = init_cell[num_models-1-m] if num_init_cell > 0 else None
      this_init_cell_deriv = init_cell_deriv[num_models-1-m] if num_init_cell > 0 else None
      this_init_hidden   = init_hidden[num_models-1-m] if num_init_cell > 0 else None
      this_init_hidden_deriv   = init_hidden_deriv[num_models-1-m] if num_init_cell > 0 else None
      this_output_deriv = output_deriv if m == num_models - 1 else None
      this_input_mr = input_mr if m == 0 else None
      this_prev_model_hidden = self.models_[num_models - 1].GetCurrentHiddenState() if m == 0 and (model.t_ - 1) > 0 else None
      this_prev_model_hidden_deriv = self.models_[num_models - 1].GetCurrentHiddenDeriv() if m == 0 and (model.t_ - 1) > 0 else None
      model.BpropAndOutp(input_frame=this_input_frame,
                         input_deriv=this_input_deriv,
                         init_cell=this_init_cell,
                         init_cell_deriv=this_init_cell_deriv,
                         init_hidden=this_init_hidden,
                         init_hidden_deriv=this_init_hidden_deriv,
                         prev_model_hidden=this_prev_model_hidden,
                         prev_model_hidden_deriv=this_prev_model_hidden_deriv,
                         input_mr=this_input_mr,
                         output_deriv=this_output_deriv)

  def Reset(self):
    for model in self.models_:
      model.Reset()

  def Update(self):
    for model in self.models_:
      model.Update()

  def GetNumModels(self):
    return self.num_models_
  
  def SetBatchSize(self, batch_size, seq_length):
    for model in self.models_:
      model.SetBatchSize(batch_size, seq_length)

  def Save(self, f):
    for model in self.models_:
      model.Save(f)

  def Load(self, f):
    for model in self.models_:
      model.Load(f)

  def Display(self):
    for m, model in enumerate(self.models_):
      model.Display(m)

  def GetParams(self):
    params_list = []
    for model in self.models_:
      params_list.extend(model.GetParams())
    return params_list

  def HasInputs(self):
    if self.num_models_ > 0:
      return self.models_[0].HasInputs()
    else:
      return False
  
  def HasOutputs(self):
    if self.num_models_ > 0:
      return self.models_[-1].HasOutputs()
    else:
      return False

  def GetInputDims(self):
    if self.num_models_ > 0:
      return self.models_[0].GetInputDims()
    else:
      return 0
  
  def GetOutputDims(self):
    if self.num_models_ > 0:
      return self.models_[-1].GetOutputDims()
    else:
      return 0

  def GetAllCurrentCellStates(self):
    return [m.GetCurrentCellState() for m in self.models_]
  
  def GetAllCurrentHiddenStates(self):
    return [m.GetCurrentHiddenState() for m in self.models_]
  
  def GetAllCurrentCellDerivs(self):
    return [m.GetCurrentCellDeriv() for m in self.models_]
  
  def GetAllCurrentHiddenDerivs(self):
    return [m.GetCurrentHiddenDeriv() for m in self.models_]
