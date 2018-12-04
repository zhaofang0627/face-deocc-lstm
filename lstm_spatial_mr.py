from util import *
import numpy as np

class LSTM(object):
  def __init__(self, lstm_config):
    self.name_ = lstm_config.name
    num_lstms = lstm_config.num_hid
    assert num_lstms  > 0
    self.num_lstms_   = num_lstms
    self.has_input_   = lstm_config.has_input
    self.has_output_  = lstm_config.has_output
    self.has_spatial_  = lstm_config.has_spatial
    self.has_spatial_dec_  = lstm_config.has_spatial_dec
    self.image_dims_ = lstm_config.image_dims
    self.image_side_ = lstm_config.image_side
    self.input_dims_  = lstm_config.input_dims
    self.input_patch_side_ = lstm_config.input_patch_side
    self.output_patch_side_ = lstm_config.output_patch_side
    self.output_dims_ = lstm_config.output_dims
    self.use_relu_    = lstm_config.use_relu
    self.input_dropprob_  = lstm_config.input_dropprob
    self.output_dropprob_ = lstm_config.output_dropprob
    self.i_ = 0
    self.j_ = 0
    self.is_imput_mr_ = 0

    self.w_dense_row_  = Param((5 * num_lstms, num_lstms), lstm_config.w_dense)
    self.w_dense_col_  = Param((5 * num_lstms, num_lstms), lstm_config.w_dense)
    self.w_diag_   = Param((num_lstms, 5), lstm_config.w_diag)
    self.b_        = Param((5 * num_lstms, 1), lstm_config.b)
    self.param_list_ = [
      ('%s:w_dense_row' % self.name_, self.w_dense_row_),
      ('%s:w_dense_col' % self.name_, self.w_dense_col_),
      ('%s:w_diag'  % self.name_, self.w_diag_),
      ('%s:b'       % self.name_, self.b_),
    ]
    if self.has_input_:
      assert self.input_dims_ > 0
      self.w_input_ = Param((5 * num_lstms, self.input_dims_), lstm_config.w_input)
      self.param_list_.append(('%s:w_input' % self.name_, self.w_input_))
      if self.has_spatial_:
        self.w_input_mr_ = Param((5 * num_lstms, self.input_dims_), lstm_config.w_input)
        self.param_list_.append(('%s:w_input_mr' % self.name_, self.w_input_mr_))
    if self.has_output_:
      assert self.output_dims_ > 0
      self.w_output_ = Param((self.output_dims_, num_lstms), lstm_config.w_output)
      self.param_list_.append(('%s:w_output' % self.name_, self.w_output_))
      self.b_output_ = Param((self.output_dims_, 1), lstm_config.b_output)
      self.param_list_.append(('%s:b_output' % self.name_, self.b_output_))

  def HasInputs(self):
    return self.has_input_

  def HasOutputs(self):
    return self.has_output_

  def GetParams(self):
    return self.param_list_

  def SetBatchSize(self, batch_size, row_length, col_length, stride=0):
    assert batch_size > 0
    assert row_length > 0
    assert col_length > 0
    self.batch_size_  = batch_size
    self.row_length_  = row_length
    self.col_length_  = col_length
    self.stride_ = stride
    seq_length = row_length * col_length
    self.gates_  = cm.empty((5 * self.num_lstms_, batch_size * seq_length))
    self.cell_   = cm.empty((self.num_lstms_, batch_size * seq_length))
    self.hidden_ = cm.empty((self.num_lstms_, batch_size * seq_length))
    self.gates_deriv_  = cm.empty_like(self.gates_)
    self.cell_deriv_   = cm.empty_like(self.cell_)
    self.hidden_deriv_ = cm.empty_like(self.hidden_)
    if self.has_input_ and self.has_spatial_:
      self.para_atten_ = cm.empty((5, batch_size))
      self.F_x_ = cm.empty((self.input_patch_side_ * self.image_side_, batch_size))
      self.F_y_ = cm.empty((self.input_patch_side_ * self.image_side_, batch_size))
      self.patches_ = cm.empty((self.input_dims_, batch_size * seq_length))
      self.patches_mr_ = cm.empty((self.input_dims_, batch_size * seq_length))
      self.patches_left_ = cm.empty((self.input_patch_side_ * self.image_side_, batch_size))
    if self.has_output_ and self.has_spatial_dec_:
      self.para_atten_hat_ = cm.empty((5, batch_size))
      self.F_x_hat_ = cm.empty((self.output_patch_side_ * self.image_side_, batch_size * seq_length))
      self.F_y_hat_ = cm.empty((self.output_patch_side_ * self.image_side_, batch_size * seq_length))
      self.output_patch_ = cm.empty((self.output_dims_, batch_size))
      self.canvas_ = cm.empty((self.image_dims_, batch_size))
      self.canvas_left_ = cm.empty((self.image_side_ * self.output_patch_side_, batch_size))
      self.output_patch_deriv_ = cm.empty_like(self.output_patch_)

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

  def Fprop(self, input_frame=None, init_cell=None, init_hidden=None, occ_pre=None, input_mr=None, output_frame=None, prev_model_hidden=None, reverse=False):
    if reverse:
      i = self.col_length_ - 1 - self.i_
      j = self.row_length_ - 1 - self.j_
    else:
      i = self.i_
      j = self.j_
    t = self.i_ * self.row_length_ + self.j_
    batch_size = self.batch_size_
    assert i >= 0
    assert j >= 0
    assert i < self.col_length_
    assert j < self.row_length_
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
          if self.has_spatial_:
            patches = self.patches_.slice(start, end)
            start_row = i * self.stride_
            start_col = j * self.stride_
            cm.get_glimpses_matrix_scan(self.F_x_, self.F_y_, start_row, start_col, self.input_patch_side_, self.image_side_)
            cm.apply_glmatrix(self.patches_left_, self.F_y_, self.para_atten_, input_frame, self.input_patch_side_, self.image_side_)
            cm.apply_glmatrix(patches, self.F_x_, self.para_atten_, self.patches_left_, self.input_patch_side_, self.image_side_)
            gates.add_dot(self.w_input_.GetW(), patches)
            if input_mr is not None:
              self.is_imput_mr_ = 1
              patches_mr = self.patches_mr_.slice(start, end)
              start_row_mr = start_row - self.input_patch_side_/2
              start_col_mr = start_col - self.input_patch_side_/2
              if start_row_mr < 0:
                start_row_mr = 0
              elif start_row_mr + self.input_patch_side_*2 > self.image_side_:
                start_row_mr = self.image_side_ - self.input_patch_side_*2
              if start_col_mr < 0:
                start_col_mr = 0
              elif start_col_mr + self.input_patch_side_*2 > self.image_side_:
                start_col_mr = self.image_side_ - self.input_patch_side_*2
              cm.get_glimpses_matrix_scan_mr3(self.F_x_, self.F_y_, start_row_mr, start_col_mr, self.input_patch_side_, self.image_side_)
              cm.apply_glmatrix(self.patches_left_, self.F_y_, self.para_atten_, input_frame, self.input_patch_side_, self.image_side_)
              cm.apply_glmatrix(patches_mr, self.F_x_, self.para_atten_, self.patches_left_, self.input_patch_side_, self.image_side_)
              gates.add_dot(self.w_input_mr_.GetW(), patches_mr)
          else:
            gates.add_dot(self.w_input_.GetW(), input_frame)
        gates.add_col_vec(self.b_.GetW())
        cm.lstm_fprop_spatial_init(gates, cell_state, hidden_state, self.w_diag_.GetW())
      else:
        cell_state.add(init_cell)
        assert init_hidden is not None
        hidden_state.add(init_hidden)
    elif self.i_ == 0:
      prev_row_start = start - batch_size
      prev_row_hidden_state = self.hidden_.slice(prev_row_start, start)
      prev_row_cell_state = self.cell_.slice(prev_row_start, start)
      if input_frame is not None:
        assert self.has_input_
        if self.has_spatial_:
          patches = self.patches_.slice(start, end)
          start_row = i * self.stride_
          start_col = j * self.stride_
          cm.get_glimpses_matrix_scan(self.F_x_, self.F_y_, start_row, start_col, self.input_patch_side_, self.image_side_)
          cm.apply_glmatrix(self.patches_left_, self.F_y_, self.para_atten_, input_frame, self.input_patch_side_, self.image_side_)
          cm.apply_glmatrix(patches, self.F_x_, self.para_atten_, self.patches_left_, self.input_patch_side_, self.image_side_)
          gates.add_dot(self.w_input_.GetW(), patches)
          if input_mr is not None:
            patches_mr = self.patches_mr_.slice(start, end)
            start_row_mr = start_row - self.input_patch_side_/2
            start_col_mr = start_col - self.input_patch_side_/2
            if start_row_mr < 0:
              start_row_mr = 0
            elif start_row_mr + self.input_patch_side_*2 > self.image_side_:
              start_row_mr = self.image_side_ - self.input_patch_side_*2
            if start_col_mr < 0:
              start_col_mr = 0
            elif start_col_mr + self.input_patch_side_*2 > self.image_side_:
              start_col_mr = self.image_side_ - self.input_patch_side_*2
            cm.get_glimpses_matrix_scan_mr3(self.F_x_, self.F_y_, start_row_mr, start_col_mr, self.input_patch_side_, self.image_side_)
            cm.apply_glmatrix(self.patches_left_, self.F_y_, self.para_atten_, input_frame, self.input_patch_side_, self.image_side_)
            cm.apply_glmatrix(patches_mr, self.F_x_, self.para_atten_, self.patches_left_, self.input_patch_side_, self.image_side_)
            gates.add_dot(self.w_input_mr_.GetW(), patches_mr)
        else:
          gates.add_dot(self.w_input_.GetW(), input_frame)
      gates.add_dot(self.w_dense_row_.GetW(), prev_row_hidden_state)
      gates.add_col_vec(self.b_.GetW())
      cm.lstm_fprop_spatial_row_init(gates, prev_row_cell_state, cell_state, hidden_state, self.w_diag_.GetW())
    elif self.j_ == 0:
      prev_col_start = (self.i_ - 1) * self.row_length_ * batch_size
      prev_col_end =  prev_col_start + batch_size
      prev_col_hidden_state = self.hidden_.slice(prev_col_start, prev_col_end)
      prev_col_cell_state = self.cell_.slice(prev_col_start, prev_col_end)
      if input_frame is not None:
        assert self.has_input_
        if self.has_spatial_:
          patches = self.patches_.slice(start, end)
          start_row = i * self.stride_
          start_col = j * self.stride_
          cm.get_glimpses_matrix_scan(self.F_x_, self.F_y_, start_row, start_col, self.input_patch_side_, self.image_side_)
          cm.apply_glmatrix(self.patches_left_, self.F_y_, self.para_atten_, input_frame, self.input_patch_side_, self.image_side_)
          cm.apply_glmatrix(patches, self.F_x_, self.para_atten_, self.patches_left_, self.input_patch_side_, self.image_side_)
          gates.add_dot(self.w_input_.GetW(), patches)
          if input_mr is not None:
            patches_mr = self.patches_mr_.slice(start, end)
            start_row_mr = start_row - self.input_patch_side_/2
            start_col_mr = start_col - self.input_patch_side_/2
            if start_row_mr < 0:
              start_row_mr = 0
            elif start_row_mr + self.input_patch_side_*2 > self.image_side_:
              start_row_mr = self.image_side_ - self.input_patch_side_*2
            if start_col_mr < 0:
              start_col_mr = 0
            elif start_col_mr + self.input_patch_side_*2 > self.image_side_:
              start_col_mr = self.image_side_ - self.input_patch_side_*2
            cm.get_glimpses_matrix_scan_mr3(self.F_x_, self.F_y_, start_row_mr, start_col_mr, self.input_patch_side_, self.image_side_)
            cm.apply_glmatrix(self.patches_left_, self.F_y_, self.para_atten_, input_frame, self.input_patch_side_, self.image_side_)
            cm.apply_glmatrix(patches_mr, self.F_x_, self.para_atten_, self.patches_left_, self.input_patch_side_, self.image_side_)
            gates.add_dot(self.w_input_mr_.GetW(), patches_mr)
        else:
          gates.add_dot(self.w_input_.GetW(), input_frame)
      gates.add_dot(self.w_dense_col_.GetW(), prev_col_hidden_state)
      gates.add_col_vec(self.b_.GetW())
      cm.lstm_fprop_spatial_col_init(gates, prev_col_cell_state, cell_state, hidden_state, self.w_diag_.GetW())
    else:
      prev_row_start = start - batch_size
      prev_row_hidden_state = self.hidden_.slice(prev_row_start, start)
      prev_row_cell_state = self.cell_.slice(prev_row_start, start)
      prev_col_start = ((self.i_ - 1) * self.row_length_ + self.j_) * batch_size
      prev_col_end =  prev_col_start + batch_size
      prev_col_hidden_state = self.hidden_.slice(prev_col_start, prev_col_end)
      prev_col_cell_state = self.cell_.slice(prev_col_start, prev_col_end)
      if input_frame is not None:
        assert self.has_input_
        if self.has_spatial_:
          patches = self.patches_.slice(start, end)
          start_row = i * self.stride_
          start_col = j * self.stride_
          cm.get_glimpses_matrix_scan(self.F_x_, self.F_y_, start_row, start_col, self.input_patch_side_, self.image_side_)
          cm.apply_glmatrix(self.patches_left_, self.F_y_, self.para_atten_, input_frame, self.input_patch_side_, self.image_side_)
          cm.apply_glmatrix(patches, self.F_x_, self.para_atten_, self.patches_left_, self.input_patch_side_, self.image_side_)
          gates.add_dot(self.w_input_.GetW(), patches)
          if input_mr is not None:
            patches_mr = self.patches_mr_.slice(start, end)
            start_row_mr = start_row - self.input_patch_side_/2
            start_col_mr = start_col - self.input_patch_side_/2
            if start_row_mr < 0:
              start_row_mr = 0
            elif start_row_mr + self.input_patch_side_*2 > self.image_side_:
              start_row_mr = self.image_side_ - self.input_patch_side_*2
            if start_col_mr < 0:
              start_col_mr = 0
            elif start_col_mr + self.input_patch_side_*2 > self.image_side_:
              start_col_mr = self.image_side_ - self.input_patch_side_*2
            cm.get_glimpses_matrix_scan_mr3(self.F_x_, self.F_y_, start_row_mr, start_col_mr, self.input_patch_side_, self.image_side_)
            cm.apply_glmatrix(self.patches_left_, self.F_y_, self.para_atten_, input_frame, self.input_patch_side_, self.image_side_)
            cm.apply_glmatrix(patches_mr, self.F_x_, self.para_atten_, self.patches_left_, self.input_patch_side_, self.image_side_)
            gates.add_dot(self.w_input_mr_.GetW(), patches_mr)
        else:
          gates.add_dot(self.w_input_.GetW(), input_frame)
      gates.add_dot(self.w_dense_row_.GetW(), prev_row_hidden_state)
      gates.add_dot(self.w_dense_col_.GetW(), prev_col_hidden_state)
      gates.add_col_vec(self.b_.GetW())
      cm.lstm_fprop_spatial(gates, prev_row_cell_state,  prev_col_cell_state, cell_state, hidden_state, self.w_diag_.GetW())

    if self.has_output_:
      assert output_frame is not None
      if self.has_spatial_dec_:
        F_x_hat = self.F_x_hat_.slice(start, end)
        F_y_hat = self.F_y_hat_.slice(start, end)
        # reconstruct reversely
        i = self.col_length_ - 1 - self.i_
        j = self.row_length_ - 1 - self.j_
        start_row = i * self.stride_
        start_col = j * self.stride_
        self.output_patch_.assign(0)
        self.output_patch_.add_dot(self.w_output_.GetW(), hidden_state)
        self.output_patch_.add_col_vec(self.b_output_.GetW())
        cm.get_glimpses_matrix_scan(F_x_hat, F_y_hat, start_row, start_col, self.output_patch_side_, self.image_side_)
        cm.apply_glmatrix_hat(self.canvas_left_, F_y_hat, self.para_atten_hat_, self.output_patch_, self.output_patch_side_, self.image_side_)
        cm.apply_glmatrix_hat(self.canvas_, F_x_hat, self.para_atten_hat_, self.canvas_left_, self.output_patch_side_, self.image_side_)
        output_frame.add(self.canvas_)
      else:
        output_frame.add_dot(self.w_output_.GetW(), hidden_state)
        output_frame.add_col_vec(self.b_output_.GetW())
    self.j_ += 1
    if self.j_ == self.row_length_ and self.i_ < self.col_length_-1:
      self.j_ = 0
      self.i_ += 1

  def BpropAndOutp(self, input_frame=None, input_deriv=None,
                   init_cell=None, init_hidden=None,
                   init_cell_deriv=None, init_hidden_deriv=None,
                   prev_model_hidden=None, prev_model_hidden_deriv=None,
                   input_mr=None, output_deriv=None):
    batch_size = self.batch_size_
    if self.j_ == 0 and self.i_ > 0:
      self.j_ = self.row_length_
      self.i_ -= 1
    self.j_ -= 1

    t = self.i_ * self.row_length_ + self.j_
    assert self.i_ >= 0
    assert self.j_ >= 0
    assert self.i_ < self.col_length_
    assert self.j_ < self.row_length_
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
      if self.has_spatial_dec_:
        F_x_hat = self.F_x_hat_.slice(start, end)
        F_y_hat = self.F_y_hat_.slice(start, end)
        cm.apply_glmatrix(self.canvas_left_, F_y_hat, self.para_atten_hat_, output_deriv, self.output_patch_side_, self.image_side_)
        cm.apply_glmatrix(self.output_patch_deriv_, F_x_hat, self.para_atten_hat_, self.canvas_left_, self.output_patch_side_, self.image_side_)
        self.w_output_.GetdW().add_dot(self.output_patch_deriv_, hidden_state.T)
        self.b_output_.GetdW().add_sums(self.output_patch_deriv_, axis=1)
        hidden_deriv.add_dot(self.w_output_.GetW().T, self.output_patch_deriv_)
      else:
        self.w_output_.GetdW().add_dot(output_deriv, hidden_state.T)
        self.b_output_.GetdW().add_sums(output_deriv, axis=1)
        hidden_deriv.add_dot(self.w_output_.GetW().T, output_deriv)

    if t == 0:
      if init_cell is None:
        assert self.has_input_
        cm.lstm_bprop_spatial_init(gates, gates_deriv, cell_state, cell_deriv, hidden_deriv, self.w_diag_.GetW())
        cm.lstm_outp_spatial_init(gates_deriv, cell_state, self.w_diag_.GetdW())
        self.b_.GetdW().add_sums(gates_deriv, axis=1)
        if self.has_spatial_:
          patches = self.patches_.slice(start, end)
          self.w_input_.GetdW().add_dot(gates_deriv, patches.T)
          if input_mr is not None:
            patches_mr = self.patches_mr_.slice(start, end)
            self.w_input_mr_.GetdW().add_dot(gates_deriv, patches_mr.T)
        else:
          self.w_input_.GetdW().add_dot(gates_deriv, input_frame.T)
        if input_deriv is not None:
          input_deriv.add_dot(self.w_input_.GetW().T, gates_deriv)
      else:
        init_hidden_deriv.add(hidden_deriv)
        init_cell_deriv.add(cell_deriv)
    elif self.i_ == 0:
      prev_row_start = start - batch_size
      prev_row_hidden_state = self.hidden_.slice(prev_row_start, start)
      prev_row_hidden_deriv = self.hidden_deriv_.slice(prev_row_start, start)
      prev_row_cell_state   = self.cell_.slice(prev_row_start, start)
      prev_row_cell_deriv   = self.cell_deriv_.slice(prev_row_start, start)
      cm.lstm_bprop_spatial_row_init(gates, gates_deriv, prev_row_cell_state, prev_row_cell_deriv,
                     cell_state, cell_deriv, hidden_deriv, self.w_diag_.GetW())
      cm.lstm_outp_spatial_row_init(gates_deriv, prev_row_cell_state, cell_state, self.w_diag_.GetdW())
      self.b_.GetdW().add_sums(gates_deriv, axis=1)
      self.w_dense_row_.GetdW().add_dot(gates_deriv, prev_row_hidden_state.T)
      prev_row_hidden_deriv.add_dot(self.w_dense_row_.GetW().T, gates_deriv)
      if input_frame is not None:
        assert self.has_input_
        if self.has_spatial_:
          patches = self.patches_.slice(start, end)
          self.w_input_.GetdW().add_dot(gates_deriv, patches.T)
          if input_mr is not None:
            patches_mr = self.patches_mr_.slice(start, end)
            self.w_input_mr_.GetdW().add_dot(gates_deriv, patches_mr.T)
        else:
          self.w_input_.GetdW().add_dot(gates_deriv, input_frame.T)
        if input_deriv is not None:
          input_deriv.add_dot(self.w_input_.GetW().T, gates_deriv)
    elif self.j_ == 0:
      prev_col_start = (self.i_ - 1) * self.row_length_ * batch_size
      prev_col_end =  prev_col_start + batch_size
      prev_col_hidden_state = self.hidden_.slice(prev_col_start, prev_col_end)
      prev_col_hidden_deriv = self.hidden_deriv_.slice(prev_col_start, prev_col_end)
      prev_col_cell_state   = self.cell_.slice(prev_col_start, prev_col_end)
      prev_col_cell_deriv   = self.cell_deriv_.slice(prev_col_start, prev_col_end)
      cm.lstm_bprop_spatial_col_init(gates, gates_deriv, prev_col_cell_state, prev_col_cell_deriv,
                     cell_state, cell_deriv, hidden_deriv, self.w_diag_.GetW())
      cm.lstm_outp_spatial_col_init(gates_deriv, prev_col_cell_state, cell_state, self.w_diag_.GetdW())
      self.b_.GetdW().add_sums(gates_deriv, axis=1)
      self.w_dense_col_.GetdW().add_dot(gates_deriv, prev_col_hidden_state.T)
      prev_col_hidden_deriv.add_dot(self.w_dense_col_.GetW().T, gates_deriv)
      if input_frame is not None:
        assert self.has_input_
        if self.has_spatial_:
          patches = self.patches_.slice(start, end)
          self.w_input_.GetdW().add_dot(gates_deriv, patches.T)
          if input_mr is not None:
            patches_mr = self.patches_mr_.slice(start, end)
            self.w_input_mr_.GetdW().add_dot(gates_deriv, patches_mr.T)
        else:
          self.w_input_.GetdW().add_dot(gates_deriv, input_frame.T)
        if input_deriv is not None:
          input_deriv.add_dot(self.w_input_.GetW().T, gates_deriv)
    else:
      prev_row_start = start - batch_size
      prev_row_hidden_state = self.hidden_.slice(prev_row_start, start)
      prev_row_hidden_deriv = self.hidden_deriv_.slice(prev_row_start, start)
      prev_row_cell_state   = self.cell_.slice(prev_row_start, start)
      prev_row_cell_deriv   = self.cell_deriv_.slice(prev_row_start, start)
      prev_col_start = ((self.i_ - 1) * self.row_length_ + self.j_) * batch_size
      prev_col_end =  prev_col_start + batch_size
      prev_col_hidden_state = self.hidden_.slice(prev_col_start, prev_col_end)
      prev_col_hidden_deriv = self.hidden_deriv_.slice(prev_col_start, prev_col_end)
      prev_col_cell_state = self.cell_.slice(prev_col_start, prev_col_end)
      prev_col_cell_deriv   = self.cell_deriv_.slice(prev_col_start, prev_col_end)
      cm.lstm_bprop_spatial(gates, gates_deriv, prev_row_cell_state, prev_row_cell_deriv, prev_col_cell_state, prev_col_cell_deriv,
                     cell_state, cell_deriv, hidden_deriv, self.w_diag_.GetW())
      cm.lstm_outp_spatial(gates_deriv, prev_row_cell_state, prev_col_cell_state, cell_state, self.w_diag_.GetdW())
      self.b_.GetdW().add_sums(gates_deriv, axis=1)
      self.w_dense_row_.GetdW().add_dot(gates_deriv, prev_row_hidden_state.T)
      prev_row_hidden_deriv.add_dot(self.w_dense_row_.GetW().T, gates_deriv)
      self.w_dense_col_.GetdW().add_dot(gates_deriv, prev_col_hidden_state.T)
      prev_col_hidden_deriv.add_dot(self.w_dense_col_.GetW().T, gates_deriv)
      if input_frame is not None:
        assert self.has_input_
        if self.has_spatial_:
          patches = self.patches_.slice(start, end)
          self.w_input_.GetdW().add_dot(gates_deriv, patches.T)
          if input_mr is not None:
            patches_mr = self.patches_mr_.slice(start, end)
            self.w_input_mr_.GetdW().add_dot(gates_deriv, patches_mr.T)
        else:
          self.w_input_.GetdW().add_dot(gates_deriv, input_frame.T)
        if input_deriv is not None:
          input_deriv.add_dot(self.w_input_.GetW().T, gates_deriv)

  def GetCurrentCellState(self):
    t = self.i_ * self.row_length_ + self.j_ - 1
    assert t >= 0 and t < self.col_length_ * self.row_length_
    batch_size = self.batch_size_
    return self.cell_.slice(t * batch_size, (t+1) * batch_size)

  def GetCurrentCellDeriv(self):
    t = self.i_ * self.row_length_ + self.j_ - 1
    assert t >= 0 and t < self.col_length_ * self.row_length_
    batch_size = self.batch_size_
    return self.cell_deriv_.slice(t * batch_size, (t+1) * batch_size)

  def GetCurrentHiddenState(self):
    t = self.i_ * self.row_length_ + self.j_ - 1
    assert t >= 0 and t < self.col_length_ * self.row_length_
    batch_size = self.batch_size_
    return self.hidden_.slice(t * batch_size, (t+1) * batch_size)
  
  def GetCurrentHiddenDeriv(self):
    t = self.i_ * self.row_length_ + self.j_ - 1
    assert t >= 0 and t < self.col_length_ * self.row_length_
    batch_size = self.batch_size_
    return self.hidden_deriv_.slice(t * batch_size, (t+1) * batch_size)

  def Update(self):
    if self.row_length_ * self.col_length_ != 1:
      self.w_dense_row_.Update()
      self.w_dense_col_.Update()
    self.w_diag_.Update()
    self.b_.Update()
    if self.has_input_:
      self.w_input_.Update()
      if self.has_spatial_:
        if self.is_imput_mr_ == 1:
          self.w_input_mr_.Update()
    if self.has_output_:
      self.w_output_.Update()
      self.b_output_.Update()

  def Reset(self):
    self.i_ = 0
    self.j_ = 0
    self.gates_.assign(0)
    self.gates_deriv_.assign(0)
    self.cell_.assign(0)
    self.cell_deriv_.assign(0)
    self.hidden_.assign(0)
    self.hidden_deriv_.assign(0)
    self.b_.GetdW().assign(0)
    self.w_dense_row_.GetdW().assign(0)
    self.w_dense_col_.GetdW().assign(0)
    self.w_diag_.GetdW().assign(0)
    if self.has_input_:
      self.w_input_.GetdW().assign(0)
      if self.has_spatial_:
        self.w_input_mr_.GetdW().assign(0)
        self.para_atten_.assign(0)
        self.F_x_.assign(0)
        self.F_y_.assign(0)
        self.patches_.assign(0)
        self.patches_left_.assign(0)
        self.patches_mr_.assign(0)
    if self.has_output_:
      self.w_output_.GetdW().assign(0)
      self.b_output_.GetdW().assign(0)
      if self.has_spatial_dec_:
        self.para_atten_hat_.assign(0)
        self.F_x_hat_.assign(0)
        self.F_y_hat_.assign(0)
        self.output_patch_.assign(0)
        self.canvas_.assign(0)
        self.canvas_left_.assign(0)
        self.output_patch_deriv_.assign(0)

  def GetInputDims(self):
    return self.input_dims_
  
  def GetOutputDims(self):
    return self.output_dims_

class LSTMStack(object):
  def __init__(self):
    self.models_ = []
    self.num_models_ = 0

  def Add(self, model):
    self.models_.append(model)
    self.num_models_ += 1

  def Fprop(self, input_frame=None, init_cell=[], init_hidden=[], occ_pre=None, input_mr=None, output_frame=None, reverse=False):
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
      this_prev_model_hidden = None
      model.Fprop(input_frame=this_input_frame,
                  init_cell=this_init_cell,
                  init_hidden=this_init_hidden,
                  occ_pre=this_occ_pre,
                  input_mr=this_input_mr,
                  output_frame=this_output_frame,
                  prev_model_hidden=this_prev_model_hidden,
                  reverse=reverse)

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
      this_prev_model_hidden = None
      this_prev_model_hidden_deriv = None
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
  
  def SetBatchSize(self, batch_size, row_length, col_length, stride):
    for model in self.models_:
      model.SetBatchSize(batch_size, row_length, col_length, stride)

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

  def GetLSTMDims(self):
    if self.num_models_ > 0:
      return self.models_[0].num_lstms_
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
