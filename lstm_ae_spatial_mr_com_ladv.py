import sys
sys.path.append('./protobuf/python/')
sys.path.append('./caffe/python/')
from data_handler import *
import lstm
import lstm_spatial_mr as lstm_spatial
# import lstm_laplace as lstm
from util import *
import Image
import h5py
import os
import cPickle
import caffe

def unpickle(filename):
    fo = open(filename, 'r')
    contents = cPickle.load(fo)
    fo.close()
    return contents

def read_mean(mean_file):
        mean_proto = caffe.io.caffe_pb2.BlobProto()
        file=open(mean_file, 'rb')
        mean_proto.ParseFromString(file.read())
        mean = np.asarray(mean_proto.data)
        return mean

class LSTMAtten(object):
  def __init__(self, model, board, board_ladv, board_sup):
    self.model_ = model
    self.board_ = board
    self.board_ladv_ = board_ladv
    self.board_sup_ = board_sup
    self.lstm_stack_enc_ = lstm_spatial.LSTMStack()
    self.lstm_stack_dec_ = lstm.LSTMStack()
    self.lstm_stack_pre_ = lstm.LSTMStack()
    model_file_sup = './data/bk20151009/part_1/bvlc_googlenet_quick_iter_231760.caffemodel'
    solver_file = './data/googlenet_ladv_solver.prototxt'
    prototxt_file_sup = './data/train_val_quick_grad.prototxt'
    mean_file = './data/bk20151009/part_1/lmdb_casia_full_part1_mean.binaryproto'
    self.cnn_solver = caffe.SGDSolver(solver_file)
    self.cnn_solver.net.copy_from(model_file_sup)
    self.cnn_net_sup = caffe.Net(prototxt_file_sup, model_file_sup, caffe.TRAIN)
    mean = read_mean(mean_file)
    mean = mean.reshape((1, 128, 128))
    self.cnn_mean_ = mean
    caffe.set_mode_gpu()
    if self.board_ladv_ == self.board_:
      caffe.set_device(self.board_ladv_)
    for l in model.lstm:
      self.lstm_stack_enc_.Add(lstm_spatial.LSTM(l))
    if model.dec_seq_length > 0:
      for l in model.lstm_dec:
        self.lstm_stack_dec_.Add(lstm.LSTM(l))
    if model.pre_seq_length > 0:
      for l in model.lstm_pre:
        self.lstm_stack_pre_.Add(lstm.LSTM(l))
    assert model.dec_seq_length > 0
    self.is_conditional_dec_ = model.dec_conditional
    if self.is_conditional_dec_ and model.dec_seq_length > 0:
      assert self.lstm_stack_dec_.HasInputs()
    self.squash_relu_ = False  #model.squash_relu
    self.squash_relu_lambda_ = 0 #model.squash_relu_lambda
    self.relu_data_ = False #model.relu_data
    self.binary_data_ = True #model.binary_data or self.squash_relu_
    self.only_occ_predict_ = model.only_occ_predict
    
    if len(model.timestamp) > 0:
      old_st = model.timestamp[-1]
      ckpt = os.path.join(model.checkpoint_dir, '%s_%s.h5' % (model.name, old_st))
      f = h5py.File(ckpt)
      self.lstm_stack_enc_.Load(f)
      if model.dec_seq_length > 0:
        self.lstm_stack_dec_.Load(f)
      if model.pre_seq_length > 0 and not self.only_occ_predict_:
        self.lstm_stack_pre_.Load(f)
      f.close()

  def Reset(self):
    self.lstm_stack_enc_.Reset()
    self.lstm_stack_dec_.Reset()
    self.lstm_stack_pre_.Reset()
    if self.dec_seq_length_ > 0:
      self.v_dec_.assign(0)
      self.v_dec_deriv_.assign(0)
    if self.pre_seq_length_ > 0:
      self.v_pre_.assign(0)
      self.v_pre_deriv_.assign(0)

  def Fprop(self, train=False):
    if self.squash_relu_:
      self.v_.apply_relu_squash(lambdaa=self.squash_relu_lambda_)
    self.Reset()
    batch_size = self.batch_size_
    enc_seq_length = self.enc_row_length_ * self.enc_col_length_
    # Fprop through encoder.
    for t in xrange(enc_seq_length):
      v = self.v_
      v_32 = self.v_32_
      # v = self.v_.slice(t * batch_size, (t+1) * batch_size)
      self.lstm_stack_enc_.Fprop(input_frame=v, input_mr=v_32)

    init_cell_states = self.lstm_stack_enc_.GetAllCurrentCellStates()
    init_hidden_states = self.lstm_stack_enc_.GetAllCurrentHiddenStates()

    # Fprop through decoder.
    if self.dec_seq_length_ > 0:
      # cm.log(self.v_.reciprocal(self.v_dec_).subtract(1)).mult(-1)
      self.lstm_stack_dec_.Fprop(init_cell=init_cell_states, init_hidden=init_hidden_states,
                                 output_frame=self.v_dec_, train=train)
      for t in xrange(1, self.dec_seq_length_):
        self.lstm_stack_dec_.Fprop(input_frame=None,
                                   output_frame=self.v_dec_,
                                   train=train)

    # Fprop through occlusion predictor.
    if self.pre_seq_length_ > 0:
      self.lstm_stack_pre_.Fprop(init_cell=init_cell_states, init_hidden=init_hidden_states,
                                 output_frame=self.v_pre_, train=train)
      for t in xrange(1, self.pre_seq_length_):
        dec_h = self.lstm_stack_dec_.models_[0].hidden_.slice(t * batch_size, (t+1) * batch_size)
        self.lstm_stack_pre_.Fprop(input_frame=dec_h,
                                   output_frame=self.v_pre_,
                                   train=train)

    if self.binary_data_:
      if self.dec_seq_length_ > 0:
        self.v_dec_.apply_sigmoid()
      if self.pre_seq_length_ > 0:
        self.v_pre_.apply_sigmoid()
    elif self.relu_data_:
      if self.dec_seq_length_ > 0:
        self.v_dec_.lower_bound(0)
      if self.pre_seq_length_ > 0:
        self.v_pre_.lower_bound(0)

  def BpropAndOutp(self):
    batch_size = self.batch_size_
    enc_seq_length = self.enc_row_length_ * self.enc_col_length_
    if self.dec_seq_length_ > 0 and not self.only_occ_predict_:
      self.v_dec_deriv_.apply_logistic_deriv(self.v_dec_)
    if self.pre_seq_length_ > 0:
      self.v_pre_deriv_.apply_logistic_deriv(self.v_pre_)

    init_cell_states = self.lstm_stack_enc_.GetAllCurrentCellStates()
    init_hidden_states = self.lstm_stack_enc_.GetAllCurrentHiddenStates()
    init_cell_derivs = self.lstm_stack_enc_.GetAllCurrentCellDerivs()
    init_hidden_derivs = self.lstm_stack_enc_.GetAllCurrentHiddenDerivs()

    # Backprop through occlusion predictor.
    if self.pre_seq_length_ > 0:
      for t in xrange(self.pre_seq_length_-1, 0, -1):
        dec_h = self.lstm_stack_dec_.models_[0].hidden_.slice(t * batch_size, (t+1) * batch_size)
        dec_h_deriv = self.lstm_stack_dec_.models_[0].hidden_deriv_.slice(t * batch_size, (t+1) * batch_size)
        self.lstm_stack_pre_.BpropAndOutp(input_frame=dec_h, input_deriv=dec_h_deriv,
                                          output_deriv=self.v_pre_deriv_)

      self.lstm_stack_pre_.BpropAndOutp(init_cell=init_cell_states,
                                        init_cell_deriv=init_cell_derivs,
                                        init_hidden=init_hidden_states,
                                        init_hidden_deriv=init_hidden_derivs,
                                        output_deriv=self.v_pre_deriv_)

    # Backprop through decoder.
    if self.dec_seq_length_ > 0 and not self.only_occ_predict_:
      for t in xrange(self.dec_seq_length_-1, 0, -1):
        self.lstm_stack_dec_.BpropAndOutp(input_frame=None,
                                          output_deriv=self.v_dec_deriv_)
        
      self.lstm_stack_dec_.BpropAndOutp(init_cell=init_cell_states,
                                        init_cell_deriv=init_cell_derivs,
                                        init_hidden=init_hidden_states,
                                        init_hidden_deriv=init_hidden_derivs,
                                        output_deriv=self.v_dec_deriv_)

    # Backprop thorough encoder.
    if not self.only_occ_predict_:
      for t in xrange(enc_seq_length-1, -1, -1):
        self.lstm_stack_enc_.BpropAndOutp(input_frame=self.v_, input_mr=self.v_32_)

  def Update(self):
    if not self.only_occ_predict_:
      self.lstm_stack_enc_.Update()
      self.lstm_stack_dec_.Update()
    self.lstm_stack_pre_.Update()

  def ComputeDeriv(self):
    batch_size = self.batch_size_
    dec = self.v_dec_
    v_target = self.t_
    deriv = self.v_dec_deriv_
    dec.subtract(v_target, target=deriv)
    deriv.divide(float(batch_size))

  def ComputeDerivComb(self):
    batch_size = self.batch_size_
    if not self.only_occ_predict_:
      deriv = self.v_dec_deriv_
      self.o_.divide(float(batch_size), target=deriv).add(self.cnn_grads_).mult(self.v_pre_)

    deriv = self.v_pre_deriv_
    self.v_dec_.subtract(self.v_, deriv)
    self.cnn_grads_.mult(deriv)
    deriv.mult(self.o_).divide(float(batch_size)).add(self.cnn_grads_)

  def GetLoss(self):
    batch_size = self.batch_size_
    dec = self.v_dec_
    v_target = self.t_
    deriv = self.v_dec_deriv_
    dec.subtract(v_target, target=deriv)

    loss_dec = 0
    if self.dec_seq_length_ > 0:
      loss_dec = 0.5 * (self.v_dec_deriv_.euclid_norm()**2) / batch_size
    return loss_dec

  def GetLossComb(self):
    batch_size = self.batch_size_
    self.v_dec_.mult(self.v_pre_, self.v_dec_deriv_)
    self.v_pre_.mult(-1, self.o_comb_).add(1).mult(self.v_).add(self.v_dec_deriv_)
    self.o_comb_.subtract(self.t_, self.o_)
    loss_comb = 0
    if self.pre_seq_length_ > 0:
      loss_comb = 0.5 * (self.o_.euclid_norm()**2) / batch_size
    return loss_comb

  def SetBatchSize(self, train_data):
    self.num_dims_ = train_data.GetDims()
    self.image_side_ = train_data.image_size_
    batch_size = train_data.GetBatchSize()
    enc_row_length = train_data.GetRowLength()
    enc_col_length = train_data.GetColLength()
    dec_seq_length = self.model_.dec_seq_length
    pre_seq_length = self.model_.pre_seq_length

    self.batch_size_ = batch_size
    self.enc_row_length_    = enc_row_length
    self.enc_col_length_    = enc_col_length
    self.dec_seq_length_    = dec_seq_length
    self.pre_seq_length_    = pre_seq_length
    self.lstm_stack_enc_.SetBatchSize(batch_size, self.enc_row_length_, self.enc_col_length_, train_data.stride_)
    self.v_ = cm.empty((self.num_dims_, batch_size))
    self.v_32_ = cm.empty((64*64, batch_size))
    self.t_ = cm.empty((self.num_dims_, batch_size))
    self.o_ = cm.empty((self.num_dims_, batch_size))
    self.o_comb_ = cm.empty((self.num_dims_, batch_size))

    if dec_seq_length > 0:
      self.lstm_stack_dec_.SetBatchSize(batch_size, dec_seq_length)
      self.v_dec_ = cm.empty((self.num_dims_, batch_size))
      self.v_dec_deriv_ = cm.empty_like(self.v_dec_)
      self.cnn_grads_ = cm.empty_like(self.v_dec_)

    if pre_seq_length > 0:
      self.lstm_stack_pre_.SetBatchSize(batch_size, pre_seq_length)
      self.v_pre_ = cm.empty((self.num_dims_, batch_size))
      self.v_pre_deriv_ = cm.empty_like(self.v_pre_)

  def Save(self, model_file):
    sys.stdout.write(' Writing model to %s' % model_file)
    f = h5py.File(model_file, 'w')
    self.lstm_stack_enc_.Save(f)
    self.lstm_stack_dec_.Save(f)
    self.lstm_stack_pre_.Save(f)
    f.close()

  def RunAndShow(self, data, output_dir=None, max_dataset_size=0):
    self.SetBatchSize(data)
    data.Reset()
    dataset_size = data.GetDatasetSize()
    if max_dataset_size > 0 and dataset_size > max_dataset_size:
      dataset_size = max_dataset_size
    batch_size = data.GetBatchSize()
    num_batches = 649 #dataset_size / batch_size
    end = True
    for ii in xrange(num_batches):
      v_cpu, v_cpu_32, t_cpu = data.GetMRBatch()
      self.v_.overwrite(v_cpu, transpose=True)
      self.v_32_.overwrite(v_cpu_32, transpose=True)
      self.Fprop()
      v_cpu = v_cpu.reshape(batch_size, data.image_size_, data.image_size_)
      t_cpu = t_cpu.reshape(batch_size, 1, data.image_size_, data.image_size_)
      rec = self.v_dec_.asarray().T.reshape(batch_size, 1, data.image_size_, data.image_size_)
      if not os.path.exists('./results'):
        os.mkdir('./results')
      avg_error = 0
      for j in xrange(batch_size):
        r_im = Image.fromarray((rec[j]*255).reshape((data.image_size_, data.image_size_)).astype(np.uint8))
        r_im.save('./results/recover_%d.jpg' % j)
        im = Image.fromarray((t_cpu[j]*255).reshape((data.image_size_, data.image_size_)).astype(np.uint8))
        im.save('./results/original_%d.jpg' % j)
        occ_im = Image.fromarray((v_cpu[j]*255).reshape((data.image_size_, data.image_size_)).astype(np.uint8))
        occ_im.save('./results/occluded_%d.jpg' % j)
        avg_error += ((rec[j]-t_cpu[j])**2).sum()
      print avg_error/batch_size
      if end:
        break

  def RunAndShowComb(self, data, output_dir=None, max_dataset_size=0):
    self.SetBatchSize(data)
    data.Reset()
    dataset_size = data.GetDatasetSize()
    if max_dataset_size > 0 and dataset_size > max_dataset_size:
      dataset_size = max_dataset_size
    batch_size = data.GetBatchSize()
    num_batches = 649 #dataset_size / batch_size
    end = True
    for ii in xrange(num_batches):
      v_cpu, v_cpu_32, t_cpu = data.GetMRBatch()
      self.v_.overwrite(v_cpu, transpose=True)
      self.v_32_.overwrite(v_cpu_32, transpose=True)
      self.Fprop()
      v_cpu = v_cpu.reshape(batch_size, data.image_size_, data.image_size_)
      t_cpu = t_cpu.reshape(batch_size, 1, data.image_size_, data.image_size_)
      self.v_dec_.mult(self.v_pre_, self.v_dec_deriv_)
      self.v_pre_.mult(-1, self.o_).add(1).mult(self.v_).add(self.v_dec_deriv_)
      o_cpu = self.o_.asarray().T.reshape(batch_size, data.image_size_, data.image_size_)
      dec_cpu = self.v_dec_.asarray().T.reshape(batch_size, data.image_size_, data.image_size_)
      pre_cpu = self.v_pre_.asarray().T.reshape(batch_size, data.image_size_, data.image_size_)
      if not os.path.exists('./results_comb'):
        os.mkdir('./results_comb')
      avg_error = 0
      for j in xrange(batch_size):
        comb_im = Image.fromarray((o_cpu[j]*255).reshape((data.image_size_, data.image_size_)).astype(np.uint8))
        comb_im.save('./results_comb/combined_%d.jpg' % j)
        rec_im = Image.fromarray((dec_cpu[j]*255).reshape((data.image_size_, data.image_size_)).astype(np.uint8))
        rec_im.save('./results_comb/recover_%d.jpg' % j)
        det_im = Image.fromarray((pre_cpu[j]*255).reshape((data.image_size_, data.image_size_)).astype(np.uint8))
        det_im.save('./results_comb/occlusion_%d.jpg' % j)
        im = Image.fromarray((t_cpu[j]*255).reshape((data.image_size_, data.image_size_)).astype(np.uint8))
        im.save('./results_comb/original_%d.jpg' % j)
        occ_im = Image.fromarray((v_cpu[j]*255).reshape((data.image_size_, data.image_size_)).astype(np.uint8))
        occ_im.save('./results_comb/occluded_%d.jpg' % j)
      if end:
        break

  def SaveImageLFW(self, data, output_dir=None, max_dataset_size=0):
    self.SetBatchSize(data)
    data.Reset()
    dataset_size = 13233 #62741 # data.GetDatasetSize()
    if max_dataset_size > 0 and dataset_size > max_dataset_size:
      dataset_size = max_dataset_size
    batch_size = data.GetBatchSize()
    num_batches = dataset_size / batch_size + 1
    end = False
    for ii in xrange(num_batches):
      print ii
      v_cpu, v_cpu_32, t_cpu, l_cpu, n_cpu = data.GetTestBatch()
      self.v_.overwrite(v_cpu, transpose=True)
      self.v_32_.overwrite(v_cpu_32, transpose=True)
      self.Fprop()
      rec = self.v_dec_.asarray().T.reshape(batch_size, data.image_size_, data.image_size_)
      if not os.path.exists('./results_lfw/recover_mr_64'):
        os.mkdir('./results_lfw/recover_mr_64')
      for j in xrange(batch_size):
        if not os.path.exists(os.path.join('./results_lfw/recover_mr_64', str(l_cpu[j][0]))):
          os.mkdir(os.path.join('./results_lfw/recover_mr_64', str(l_cpu[j][0])))
        rec_im = Image.fromarray((rec[j]*255).reshape((data.image_size_, data.image_size_)).astype(np.uint8))
        rec_im.save(os.path.join('./results_lfw/recover_mr_64', str(l_cpu[j][0]), str(n_cpu[j][0])+'.jpg'))
      if end:
        break

  def SaveImageLFWComb(self, data, output_dir=None, max_dataset_size=0):
    self.SetBatchSize(data)
    data.Reset()
    dataset_size = 13233 # data.GetDatasetSize()
    if max_dataset_size > 0 and dataset_size > max_dataset_size:
      dataset_size = max_dataset_size
    batch_size = data.GetBatchSize()
    num_batches = dataset_size / batch_size + 1
    end = False
    for ii in xrange(num_batches):
      print ii
      v_cpu, v_cpu_32, t_cpu, l_cpu, n_cpu = data.GetTestBatch()
      self.v_.overwrite(v_cpu, transpose=True)
      self.v_32_.overwrite(v_cpu_32, transpose=True)
      self.Fprop()
      self.v_dec_.mult(self.v_pre_, self.v_dec_deriv_)
      self.v_pre_.mult(-1, self.o_).add(1).mult(self.v_).add(self.v_dec_deriv_)
      o_cpu = self.o_.asarray().T.reshape(batch_size, data.image_size_, data.image_size_)
      if not os.path.exists('./results_lfw/combined_only_ladv'):
        os.mkdir('./results_lfw/combined_only_ladv')
      for j in xrange(batch_size):
        if not os.path.exists(os.path.join('./results_lfw/combined_only_ladv', str(l_cpu[j][0]))):
          os.mkdir(os.path.join('./results_lfw/combined_only_ladv', str(l_cpu[j][0])))
        comb_im = Image.fromarray((o_cpu[j]*255).reshape((data.image_size_, data.image_size_)).astype(np.uint8))
        comb_im.save(os.path.join('./results_lfw/combined_only_ladv', str(l_cpu[j][0]), str(n_cpu[j][0])+'.jpg'))
      if end:
        break

  def SaveImage50P(self, data, output_dir=None, max_dataset_size=0):
    self.SetBatchSize(data)
    data.Reset()
    dataset_size = data.GetDatasetSize()
    if max_dataset_size > 0 and dataset_size > max_dataset_size:
      dataset_size = max_dataset_size
    batch_size = data.GetBatchSize()
    num_batches = dataset_size / batch_size + 1
    end = False
    for ii in xrange(num_batches):
      print ii
      v_cpu, v_cpu_32, t_cpu, l_cpu, n_cpu = data.GetTestBatch()
      self.v_.overwrite(v_cpu, transpose=True)
      self.v_32_.overwrite(v_cpu_32, transpose=True)
      self.Fprop()
      rec = self.v_dec_.asarray().T.reshape(batch_size, data.image_size_, data.image_size_)
      if not os.path.exists('./results_50p/recover_mr_64'):
        os.mkdir('./results_50p/recover_mr_64')
      for j in xrange(batch_size):
        if not os.path.exists(os.path.join('./results_50p/recover_mr_64', str(l_cpu[j][0]))):
          os.mkdir(os.path.join('./results_50p/recover_mr_64', str(l_cpu[j][0])))
        rec_im = Image.fromarray((rec[j]*255).reshape((data.image_size_, data.image_size_)).astype(np.uint8))
        rec_im.save(os.path.join('./results_50p/recover_mr_64', str(l_cpu[j][0]), str(n_cpu[j][0])+'.jpg'))
      if end:
        break

  def SaveImage50PComb(self, data, output_dir=None, max_dataset_size=0):
    self.SetBatchSize(data)
    data.Reset()
    dataset_size = data.GetDatasetSize()
    if max_dataset_size > 0 and dataset_size > max_dataset_size:
      dataset_size = max_dataset_size
    batch_size = data.GetBatchSize()
    num_batches = dataset_size / batch_size + 1
    end = False
    for ii in xrange(num_batches):
      print ii
      v_cpu, v_cpu_32, t_cpu, l_cpu, n_cpu = data.GetTestBatch()
      self.v_.overwrite(v_cpu, transpose=True)
      self.v_32_.overwrite(v_cpu_32, transpose=True)
      self.Fprop()
      self.v_dec_.mult(self.v_pre_, self.v_dec_deriv_)
      self.v_pre_.mult(-1, self.o_).add(1).mult(self.v_).add(self.v_dec_deriv_)
      o_cpu = self.o_.asarray().T.reshape(batch_size, data.image_size_, data.image_size_)
      if not os.path.exists('./results_50p/original'):
        os.mkdir('./results_50p/original')
      if not os.path.exists('./results_50p/occluded'):
        os.mkdir('./results_50p/occluded')
      if not os.path.exists('./results_50p/combined_com_ladv'):
        os.mkdir('./results_50p/combined_com_ladv')
      for j in xrange(batch_size):
        if not os.path.exists(os.path.join('./results_50p/original', str(l_cpu[j][0]))):
          os.mkdir(os.path.join('./results_50p/original', str(l_cpu[j][0])))
        ori_im = Image.fromarray((t_cpu[j]*255).reshape((data.image_size_, data.image_size_)).astype(np.uint8))
        ori_im.save(os.path.join('./results_50p/original', str(l_cpu[j][0]), str(n_cpu[j][0])+'.jpg'))
        if not os.path.exists(os.path.join('./results_50p/occluded', str(l_cpu[j][0]))):
          os.mkdir(os.path.join('./results_50p/occluded', str(l_cpu[j][0])))
        occ_im = Image.fromarray((v_cpu[j]*255).reshape((data.image_size_, data.image_size_)).astype(np.uint8))
        occ_im.save(os.path.join('./results_50p/occluded', str(l_cpu[j][0]), str(n_cpu[j][0])+'.jpg'))
        if not os.path.exists(os.path.join('./results_50p/combined_com_ladv', str(l_cpu[j][0]))):
          os.mkdir(os.path.join('./results_50p/combined_com_ladv', str(l_cpu[j][0])))
        comb_im = Image.fromarray((o_cpu[j]*255).reshape((data.image_size_, data.image_size_)).astype(np.uint8))
        comb_im.save(os.path.join('./results_50p/combined_com_ladv', str(l_cpu[j][0]), str(n_cpu[j][0])+'.jpg'))
      if end:
        break

  def Train(self, train_data, valid_data=None):
    # Timestamp the model that we are training.
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
    model_file = os.path.join(self.model_.checkpoint_dir, '%s_%s' % (self.model_.name, st))
    self.model_.timestamp.append(st)
    WritePbtxt(self.model_, '%s.pbtxt' % model_file)
    print 'Model saved at %s.pbtxt' % model_file
   
    self.SetBatchSize(train_data)

    loss_comb = 0
    loss_sup = 0
    loss_dec = 0

    print_after = self.model_.print_after
    validate_after = self.model_.validate_after
    validate = validate_after > 0 and valid_data is not None
    save_after = self.model_.save_after
    save = save_after > 0
    display_after = self.model_.display_after
    display = display_after > 0

    for ii in xrange(1, self.model_.max_iters + 1):
      newline = False
      sys.stdout.write('\rStep %d' % ii)
      sys.stdout.flush()
      v_cpu, v_cpu_32, t_cpu, l_cpu_sup = train_data.GetLabeledMRBatch()
      l_cpu = np.where(l_cpu_sup==1)[1]
      l_cpu[...] = 0
      l_o_cpu = l_cpu + train_data.num_person
      l_o_cpu[...] = 1
      self.v_.overwrite(v_cpu, transpose=True)
      self.v_32_.overwrite(v_cpu_32, transpose=True)
      self.t_.overwrite(t_cpu, transpose=True)
      self.Fprop(train=True)

      # Compute Performance.
      if self.pre_seq_length_ > 0:
        this_loss_comb = self.GetLossComb()
        loss_comb += this_loss_comb
        # Discriminator
        o_cpu = self.o_comb_.asarray().T
        cnn_input = np.vstack((t_cpu, o_cpu)).reshape(self.batch_size_*2, 1, self.image_side_, self.image_side_)
        cnn_input *= 255
        cnn_input -= self.cnn_mean_
        if self.board_sup_ != self.board_:
          caffe.set_device(self.board_sup_)
        self.cnn_net_sup.blobs['data'].data[...] = cnn_input[self.batch_size_:]
        self.cnn_net_sup.blobs['label'].data[...] = np.where(l_cpu_sup==1)[1].reshape(self.batch_size_, 1, 1, 1)
        cnn_outs_sup = self.cnn_net_sup.forward()
        loss_sup += cnn_outs_sup['loss1/loss1']*0.3+cnn_outs_sup['loss2/loss1']*0.3+cnn_outs_sup['loss3/loss3']
        cnn_grads_sup = self.cnn_net_sup.backward()
        cnn_grad_sup = cnn_grads_sup['data']
        if self.board_ladv_ != self.board_sup_:
          caffe.set_device(self.board_ladv_)
        self.cnn_solver.net.blobs['data'].data[...] = np.concatenate((cnn_input[self.batch_size_:], cnn_input[self.batch_size_:]), axis=0)
        self.cnn_solver.net.blobs['label'].data[...] = np.hstack((l_cpu, l_cpu)).reshape(self.batch_size_*2, 1, 1, 1)
        cnn_outs = self.cnn_solver.net.forward()
        cnn_grads = self.cnn_solver.net.backward()
        cnn_grad = cnn_grads['data'][:self.batch_size_]
        cnn_grad += cnn_grad_sup
        cnn_grad *= 255
        self.cnn_grads_.overwrite(cnn_grad.reshape(self.batch_size_, self.num_dims_), transpose=True)
        self.cnn_solver.net.blobs['data'].data[...] = cnn_input
        self.cnn_solver.net.blobs['label'].data[...] = np.hstack((l_cpu, l_o_cpu)).reshape(self.batch_size_*2, 1, 1, 1)
        self.cnn_solver.step(1)
        if self.board_ladv_ != self.board_:
          cm.cuda_set_device(self.board_)
        # Compute Derivative
        self.ComputeDerivComb()
        if ii % print_after == 0:
          loss_comb /= print_after
          loss_sup /= print_after
          sys.stdout.write(' Comb %.5f Cnn_Sup %.5f' % (loss_comb, loss_sup))
          loss_comb = 0
          loss_sup = 0
          newline = True
      else:
        this_loss_dec = self.GetLoss()
        loss_dec += this_loss_dec
        self.ComputeDeriv()
        if ii % print_after == 0:
          loss_dec /= print_after
          sys.stdout.write(' Dec %.5f' % loss_dec)
          loss_dec = 0
          newline = True

      self.BpropAndOutp()
      self.Update()

      if validate and ii % validate_after == 0:
        valid_loss_dec, valid_loss_fut = self.Validate(valid_data)
        sys.stdout.write(' VDec %.5f VFut %.5f' % (valid_loss_dec, valid_loss_fut))
        newline = True

      if save and ii % save_after == 0:
        self.Save('%s.h5' % model_file)
        self.cnn_solver.net.save(str('%s.caffemodel' % model_file))
        print ' %s.caffemodel' % model_file
      if newline:
        sys.stdout.write('\n')

    sys.stdout.write('\n')

def main():
  board = int(sys.argv[4])
  cm.cuda_set_device(int(board))
  cm.cublas_init()
  board_cnn = int(sys.argv[5])
  board_sup = int(sys.argv[6])
  print 'Using board', board
  cm.CUDAMatrix.init_random(42)
  np.random.seed(42)

  model      = ReadModelProto(sys.argv[1])
  lstm_autoencoder = LSTMAtten(model, board, board_cnn, board_sup)
  train_data = ChooseDataHandler(ReadDataProto(sys.argv[2]))
  # if len(sys.argv) > 4:
  #   valid_data = ChooseDataHandler(ReadDataProto(sys.argv[4]))
  # else:
  valid_data = None
  if sys.argv[3] == 'test':
    lstm_autoencoder.RunAndShow(train_data)
  elif sys.argv[3] == 'test_comb':
    lstm_autoencoder.RunAndShowComb(train_data)
  elif sys.argv[3] == 'save_lfw':
    lstm_autoencoder.SaveImageLFW(train_data)
  elif sys.argv[3] == 'save_lfw_comb':
    lstm_autoencoder.SaveImageLFWComb(train_data)
  elif sys.argv[3] == 'save_50p':
    lstm_autoencoder.SaveImage50P(train_data)
  elif sys.argv[3] == 'save_50p_comb':
    lstm_autoencoder.SaveImage50PComb(train_data)
  else:
    lstm_autoencoder.Train(train_data, valid_data)
  """
  lstm_autoencoder.GradCheck()
  """

if __name__ == '__main__':
  # board = LockGPU()
  main()
  # FreeGPU(board)
