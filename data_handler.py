import numpy as np
import config_pb2
from google.protobuf import text_format
import h5py
import sys
import math
import json
import caffe
import copy
import time
# from threading import Thread

def ReadDataProto(fname):
  data_pb = config_pb2.Data()
  with open(fname, 'r') as pbtxt:
    text_format.Merge(pbtxt.read(), data_pb)
  return data_pb

def ChooseDataHandler(data_pb):
  if data_pb.dataset_type == config_pb2.Data.LABELLED:
    return CPUSequenceBuffer(data_pb)
  elif data_pb.dataset_type == config_pb2.Data.BOUNCING_MNIST:
    return BouncingMNISTDataHandler(data_pb)
  elif data_pb.dataset_type == config_pb2.Data.OCC_FACE:
    return OccFaceDataHandler(data_pb)
  elif data_pb.dataset_type == config_pb2.Data.OCC_FACE_128:
    return OccFaceData128Handler(data_pb)
  elif data_pb.dataset_type == config_pb2.Data.LFW128:
    return LFW128DataHandler(data_pb)
  elif data_pb.dataset_type == config_pb2.Data.OCC_FACE_STN:
    return OccFaceSTNDataHandler(data_pb)
  elif data_pb.dataset_type == config_pb2.Data.MNIST:
    return MNISTDataHandler(data_pb)
  else:
    raise Exception('Unknown DatasetType.')

def GetBoundaries(filename):
  boundaries = []
  num_frames = []
  start = 0
  for line in open(filename):
    num_f = int(line.strip())
    num_frames.append(num_f)
    end = start + num_f
    boundaries.append((start, end))
    start = end
  return boundaries, num_frames

def GetVideoIds(filename):
  video_ids = []
  if filename != '':
    for line in open(filename):
      video_ids.append(int(line.strip()))
  return video_ids

def GetLabels(filename):
  labels = []
  if filename != '':
    for line in open(filename):
      labels.append(int(line.strip()))
  return labels


class CPUSequenceBuffer(object):
  def __init__(self, data_pb):
    self.data_ = h5py.File(data_pb.data_file)[data_pb.dataset_name]
    self.num_dims_ = self.data_.shape[1]
    self.video_boundaries_, num_frames = GetBoundaries(data_pb.num_frames_file)
    self.labels_ = GetLabels(data_pb.labels_file)
    video_ids = GetVideoIds(data_pb.video_ids_file)
    if len(video_ids) == 0:
      video_ids = range(len(num_frames))
    self.seq_length_ = data_pb.num_frames
    self.seq_stride_ = data_pb.stride
    self.randomize_ = data_pb.randomize
    self.batch_size_ = data_pb.batch_size  # Not used.
    self.sampling_factor_ = data_pb.sampling_factor
    self.dataset_size_ = 0
    total_frames = 0
    self.video_ids_ = []
    for vid_id in video_ids:
      num_f = num_frames[vid_id]
      if num_f >= self.seq_length_:
        total_frames += num_f
        self.dataset_size_ += (num_f - self.seq_length_) / self.seq_stride_ + 1
        self.video_ids_.append(vid_id)
    self.num_videos_ = len(self.video_ids_)
    self.buffer_size_ = (data_pb.cpu_buffer_size * 1024 * 1024) / (4 * self.num_dims_)
    if total_frames <= self.buffer_size_:
      self.load_once_ = True
      self.buffer_size_ = total_frames
    else:
      self.load_once_ = False
    print 'Total frames: %d Buffer Size %d' % (total_frames, self.buffer_size_)
    self.buffer_ = np.zeros((self.buffer_size_, self.num_dims_), dtype=np.float32)
    self.labels_buffer_ = np.zeros(self.buffer_size_, dtype=np.int32)
    self.loaded_once_ = False
    self.Reset()


  def GetAccuracy(self, preds):
    s = 0
    #num_classes = preds.shape[1]
    #predictions = np.zeros((self.num_videos_, num_classes), dtype=np.float32)
    correct = 0
    for i, v in enumerate(self.video_ids_):
      true_label = self.labels_[v]
      start, end = self.video_boundaries_[v]
      num_preds = (end - start - self.seq_length_) / self.seq_stride_ + 1
      #predictions[i, :] = preds[s:s+num_preds].mean(axis=0)
      p = preds[s:s+num_preds, :].mean(axis=0)
      if p.argmax() == true_label:
        correct += 1
      s += num_preds
    acc = float(correct) / self.num_videos_
    return acc

  def Reset(self):
    self.buffer_ind_ = 0
    self.carry_over_start_ = 0
    self.carry_over_size_ = 0
    self.carry_over_label_ = 0
    self.video_ind_ = 0
    self.buffer_start_indices_ = []
    self.batch_video_id_ = 0
    self.num_batches_in_buffer_ = 0

  def GetDataSetSize(self):
    return self.dataset_size_

  def GetSeqLength(self):
    return self.seq_length_

  def GetBatchSize(self):
    return self.batch_size_

  def GetDims(self):
    return self.num_dims_

  def RefillBuffer(self):
    # Fill carry over
    s = 0
    self.buffer_start_indices_ = []
    if self.carry_over_size_ > 0:
      #print 'Carry over size', self.carry_over_size_
      if self.carry_over_size_ > self.buffer_size_:
        self.buffer_[:, :] = \
            self.data_[self.carry_over_start_:self.carry_over_start_+self.buffer_size_, :]
        last_loc = ((self.buffer_size_ - self.seq_length_) / self.seq_stride_ + 1) * self.seq_stride_
        self.carry_over_size_  -= last_loc
        self.carry_over_start_ += last_loc
        self.labels_buffer_[:] = self.carry_over_label_
        if self.carry_over_size_ < self.seq_length_:
          self.carry_over_size_ = 0
          self.video_ind_ = (self.video_ind_ + 1) % self.num_videos_
        s = self.buffer_size_
      else:
        self.buffer_[:self.carry_over_size_, :] = \
            self.data_[self.carry_over_start_:self.carry_over_start_+self.carry_over_size_, :]
        self.labels_buffer_[:self.carry_over_size_] = self.carry_over_label_
        s = self.carry_over_size_
        self.carry_over_size_ = 0
        self.video_ind_ = (self.video_ind_ + 1) % self.num_videos_
      self.buffer_start_indices_.extend(range(0, s-self.seq_length_+1, self.seq_stride_))
    # Fill more videos.
    while s < self.buffer_size_:
      if self.video_ind_ == 0 and self.randomize_:
        np.random.shuffle(self.video_ids_)
      video_id = self.video_ids_[self.video_ind_]
      start, end = self.video_boundaries_[video_id]
      label = self.labels_[video_id]
      #print 'Video index', self.video_ind_, video_id, start, end, end-start
      if s + end - start <= self.buffer_size_:
        if end - start >= self.seq_length_:  # Use only those videos which have at least seq length frames.
          self.buffer_[s:s+end-start, :] = self.data_[start:end]
          self.buffer_start_indices_.extend(range(s, s+end-start-self.seq_length_+1, self.seq_stride_))
          self.labels_buffer_[s:s+end-start] = label
          s += end-start
        else:
          raise Exception('This should not happen')
        self.video_ind_ = (self.video_ind_ + 1) % self.num_videos_
      else:
        space_left = self.buffer_size_ - s
        if space_left < self.seq_length_:
          self.carry_over_start_ = start
          self.carry_over_size_ = end - start
        else:
          self.buffer_[s:s+space_left, :] = self.data_[start:start+space_left, :]
          self.buffer_start_indices_.extend(range(s, self.buffer_size_-self.seq_length_+1, self.seq_stride_))
          self.labels_buffer_[s:s+space_left] = label
          self.carry_over_start_ = start + ((space_left - self.seq_length_) / self.seq_stride_ + 1) * self.seq_stride_
          self.carry_over_size_ = end - self.carry_over_start_
          if self.carry_over_size_ < self.seq_length_:
            self.carry_over_size_ = 0
        s = self.buffer_size_

  def GetBatch(self, batch_size):
    seq_length = self.seq_length_
    num_dims = self.num_dims_
    batch = np.zeros((seq_length, batch_size, num_dims), dtype=np.float32)
    labels = np.zeros(batch_size, dtype=np.int32)
    for i in xrange(batch_size):
      if self.batch_video_id_ == 0:
        if not (self.load_once_ and self.loaded_once_):
          self.RefillBuffer()
          self.loaded_once_ = True
          self.num_batches_in_buffer_ = len(self.buffer_start_indices_)
        if self.randomize_:
          np.random.shuffle(self.buffer_start_indices_)
      start = self.buffer_start_indices_[self.batch_video_id_]
      batch[:, i, :] = self.buffer_[start:start+seq_length, :]
      labels[i] = self.labels_buffer_[start]
      self.batch_video_id_ += self.sampling_factor_
      if self.batch_video_id_ >= self.num_batches_in_buffer_:
        self.batch_video_id_ = 0
    return batch, labels

class BouncingMNISTDataHandler(object):
  def __init__(self, data_pb):
    self.seq_length_ = data_pb.num_frames
    self.batch_size_ = data_pb.batch_size
    self.image_size_ = data_pb.image_size
    self.num_digits_ = data_pb.num_digits
    self.step_length_ = data_pb.step_length
    self.dataset_size_ = 10000  # The dataset is really infinite. This is just for validation.
    self.digit_size_ = 28
    self.frame_size_ = self.image_size_ ** 2
    f = h5py.File(data_pb.data_file)
    self.data_ = f[data_pb.dataset_name].value.reshape(-1, 28, 28)
    f.close()
    self.indices_ = np.arange(self.data_.shape[0])
    self.row_ = 0
    np.random.shuffle(self.indices_)

  def GetBatchSize(self):
    return self.batch_size_

  def GetDims(self):
    return self.frame_size_

  def GetDatasetSize(self):
    return self.dataset_size_

  def GetSeqLength(self):
    return self.seq_length_

  def Reset(self):
    self.row_ = 0

  def GetRandomTrajectory(self, batch_size):
    length = self.seq_length_
    canvas_size = self.image_size_ - self.digit_size_
    
    # Initial position uniform random inside the box.
    y = np.random.rand(batch_size)
    x = np.random.rand(batch_size)

    # Choose a random velocity.
    theta = np.random.rand(batch_size) * 2 * np.pi
    v_y = np.sin(theta)
    v_x = np.cos(theta)

    start_y = np.zeros((length, batch_size))
    start_x = np.zeros((length, batch_size))
    for i in xrange(length):
      # Take a step along velocity.
      y += v_y * self.step_length_
      x += v_x * self.step_length_

      # Bounce off edges.
      for j in xrange(batch_size):
        if x[j] <= 0:
          x[j] = 0
          v_x[j] = -v_x[j]
        if x[j] >= 1.0:
          x[j] = 1.0
          v_x[j] = -v_x[j]
        if y[j] <= 0:
          y[j] = 0
          v_y[j] = -v_y[j]
        if y[j] >= 1.0:
          y[j] = 1.0
          v_y[j] = -v_y[j]
      start_y[i, :] = y
      start_x[i, :] = x

    # Scale to the size of the canvas.
    start_y = (canvas_size * start_y).astype(np.int32)
    start_x = (canvas_size * start_x).astype(np.int32)
    return start_y, start_x

  def Overlap(self, a, b):
    """ Put b on top of a."""
    return np.maximum(a, b)
    #return b

  def GetBatch(self, verbose=False):
    start_y, start_x = self.GetRandomTrajectory(self.batch_size_ * self.num_digits_)
    data = np.zeros((self.seq_length_, self.batch_size_, self.image_size_, self.image_size_), dtype=np.float32)
    for j in xrange(self.batch_size_):
      for n in xrange(self.num_digits_):
        ind = self.indices_[self.row_]
        self.row_ += 1
        if self.row_ == self.data_.shape[0]:
          self.row_ = 0
          np.random.shuffle(self.indices_)
        digit_image = self.data_[ind, :, :]
        for i in xrange(self.seq_length_):
          top    = start_y[i, j * self.num_digits_ + n]
          left   = start_x[i, j * self.num_digits_ + n]
          bottom = top  + self.digit_size_
          right  = left + self.digit_size_
          data[i, j, top:bottom, left:right] = self.Overlap(data[i, j, top:bottom, left:right], digit_image)
    return data.reshape(self.seq_length_ * self.batch_size_, -1), None

  def DisplayData(self, data, rec=None, fut=None, fig=1, case_id=0, output_file=None):
    name, ext = os.path.splitext(output_file)
    output_file1 = '%s_original%s' % (name, ext)
    output_file2 = '%s_recon%s' % (name, ext)
    data = data[case_id, :].reshape(-1, self.image_size_, self.image_size_)
    if rec is not None:
      rec = rec[case_id, :].reshape(-1, self.image_size_, self.image_size_)
      enc_seq_length = rec.shape[0]
    if fut is not None:
      fut = fut[case_id, :].reshape(-1, self.image_size_, self.image_size_)
      if rec is None:
        enc_seq_length = self.seq_length_ - fut.shape[0]
      else:
        assert enc_seq_length == self.seq_length_ - fut.shape[0]
    num_rows = 1
    plt.figure(2*fig, figsize=(20, 1))
    plt.clf()
    for i in xrange(self.seq_length_):
      plt.subplot(num_rows, self.seq_length_, i+1)
      plt.imshow(data[i, :, :], cmap=plt.cm.gray, interpolation="nearest")
      plt.axis('off')
    plt.draw()
    if output_file1 is not None:
      print output_file1
      plt.savefig(output_file1, bbox_inches='tight')

    plt.figure(2*fig+1, figsize=(20, 1))
    plt.clf()
    for i in xrange(self.seq_length_):
      if rec is not None and i < enc_seq_length:
        plt.subplot(num_rows, self.seq_length_, i + 1)
        plt.imshow(rec[rec.shape[0] - i - 1, :, :], cmap=plt.cm.gray, interpolation="nearest")
      if fut is not None and i >= enc_seq_length:
        plt.subplot(num_rows, self.seq_length_, i + 1)
        plt.imshow(fut[i - enc_seq_length, :, :], cmap=plt.cm.gray, interpolation="nearest")
      plt.axis('off')
    plt.draw()
    if output_file2 is not None:
      print output_file2
      plt.savefig(output_file2, bbox_inches='tight')
    else:
      plt.pause(0.1)

class OccFaceDataHandler(object):
  def __init__(self, data_pb):
    self.seq_length_ = data_pb.num_frames
    self.batch_size_ = data_pb.batch_size
    self.image_size_ = data_pb.image_size
    if self.batch_size_ == 256:
      self.dataset_size_ = 585 * 2 * self.batch_size_  # The dataset is really infinite. This is just for validation.
    else:
      self.dataset_size_ = 585 * self.batch_size_
    self.frame_size_ = self.image_size_ ** 2
    f = h5py.File(data_pb.data_file)
    self.data_ = f[data_pb.dataset_name].value.reshape(-1, self.frame_size_)
    self.target_ = f[data_pb.target_name].value.reshape(-1, self.frame_size_)
    f.close()
    self.indices_ = np.arange(self.dataset_size_)
    self.row_ = 0
    # np.random.shuffle(self.indices_)

  def GetBatchSize(self):
    return self.batch_size_

  def GetDims(self):
    return self.frame_size_

  def GetDatasetSize(self):
    return self.dataset_size_

  def GetSeqLength(self):
    return self.seq_length_

  def Reset(self):
    self.row_ = 0

  def Overlap(self, a, b):
    """ Put b on top of a."""
    return np.maximum(a, b)
    #return b

  def GetBatch(self, verbose=False):
    data = np.zeros((self.batch_size_, self.frame_size_), dtype=np.float32)
    target = np.zeros((self.batch_size_, self.frame_size_), dtype=np.float32)
    for j in xrange(self.batch_size_):
      ind = self.indices_[self.row_]
      self.row_ += 1
      if self.row_ == self.dataset_size_:
        self.row_ = 0
        np.random.shuffle(self.indices_)
      data[j, :] = self.data_[ind, :]
      target[j, :] = self.target_[ind, :]
    return data, target

  def GetTestBatch(self, verbose=False):
    data = np.zeros((self.batch_size_, self.frame_size_), dtype=np.float32)
    target = np.zeros((self.batch_size_, self.frame_size_), dtype=np.float32)
    for j in xrange(self.batch_size_):
      # ind = self.indices_[self.row_]
      # self.row_ += 1
      # if self.row_ == self.dataset_size_:
      #   self.row_ = 0
      #   np.random.shuffle(self.indices_)
      data[j, :] = self.data_[self.dataset_size_+j, :]
      target[j, :] = self.target_[self.dataset_size_+j, :]
    return data, target

class OccFaceData128Handler(object):
  def __init__(self, data_pb):
    self.data_pb = data_pb
    self.seq_length_ = data_pb.num_frames
    self.row_length_ = data_pb.row_length
    self.col_length_ = data_pb.col_length
    self.stride_ = data_pb.stride
    self.stride_dec_ = data_pb.stride_dec
    self.batch_size_ = data_pb.batch_size
    self.image_size_ = data_pb.image_size
    self.num_person = 10575
    if self.batch_size_ == 128:
      self.dataset_size_ = 743 * 4 * self.batch_size_
    elif self.batch_size_ == 256:
      self.dataset_size_ = 743 * 2 * self.batch_size_
    else:
      self.dataset_size_ = 743 * self.batch_size_
    self.frame_size_ = self.image_size_ ** 2
    self.frame_pad_size_ = (self.image_size_ + 32) ** 2
    self.df = h5py.File(data_pb.data_file)
    self.indices_ = np.arange(self.dataset_size_)
    self.row_ = 0
    np.random.shuffle(self.indices_)

  def GetBatchSize(self):
    return self.batch_size_

  def GetDims(self):
    return self.frame_size_

  def GetPadDims(self):
    return self.frame_pad_size_

  def GetDatasetSize(self):
    return self.dataset_size_

  def GetSeqLength(self):
    return self.seq_length_

  def GetRowLength(self):
    return self.row_length_

  def GetColLength(self):
    return self.col_length_

  def Reset(self):
    self.row_ = 0

  def Overlap(self, a, b):
    """ Put b on top of a."""
    return np.maximum(a, b)
    #return b

  def GetBatch(self, verbose=False):
    data = self.df[self.data_pb.dataset_name][self.row_*self.batch_size_:(self.row_+1)*self.batch_size_].reshape(-1, self.frame_size_)
    target = self.df[self.data_pb.target_name][self.row_*self.batch_size_:(self.row_+1)*self.batch_size_].reshape(-1, self.frame_size_)
    self.row_ += 1
    if self.row_ == self.dataset_size_/self.batch_size_:
      self.row_ = 0
    return data, target

  def GetVaeBatch(self, verbose=False):
    data = self.df[self.data_pb.dataset_name][self.row_*self.batch_size_:(self.row_+1)*self.batch_size_].reshape(-1, 1, self.image_size_, self.image_size_)
    target = self.df[self.data_pb.target_name][self.row_*self.batch_size_:(self.row_+1)*self.batch_size_].reshape(-1, 1, self.image_size_, self.image_size_)
    self.row_ += 1
    if self.row_ == self.dataset_size_/self.batch_size_:
      self.row_ = 0
    return data, target

  def GetLabeledBatch(self, verbose=False):
    data = self.df[self.data_pb.dataset_name][self.row_*self.batch_size_:(self.row_+1)*self.batch_size_].reshape(-1, self.frame_size_)
    target = self.df[self.data_pb.target_name][self.row_*self.batch_size_:(self.row_+1)*self.batch_size_].reshape(-1, self.frame_size_)
    labels = self.df[self.data_pb.labels_name][self.row_*self.batch_size_:(self.row_+1)*self.batch_size_]
    self.row_ += 1
    if self.row_ == self.dataset_size_/self.batch_size_:
      self.row_ = 0
    return data, target, labels

  def GetTestBatch(self, verbose=False):
    data = self.df[self.data_pb.dataset_name][(self.dataset_size_/self.batch_size_)*self.batch_size_:(self.dataset_size_/self.batch_size_+1)*self.batch_size_].reshape(-1, self.frame_size_)
    target = self.df[self.data_pb.target_name][(self.dataset_size_/self.batch_size_)*self.batch_size_:(self.dataset_size_/self.batch_size_+1)*self.batch_size_].reshape(-1, self.frame_size_)
    return data, target

  def GetScrubBatch(self, verbose=False):
    data = self.df[self.data_pb.dataset_name][self.row_*self.batch_size_:(self.row_+1)*self.batch_size_].reshape(-1, self.frame_size_)
    target = self.df[self.data_pb.target_name][self.row_*self.batch_size_:(self.row_+1)*self.batch_size_].reshape(-1, self.frame_size_)
    labels = self.df[self.data_pb.labels_name][self.row_*self.batch_size_:(self.row_+1)*self.batch_size_].reshape(-1, 1)
    self.row_ += 1
    if self.row_ == 62741/self.batch_size_:
      self.row_ = 0
    return data, target, labels

  def GetMRBatch(self, verbose=False):
    data = self.df[self.data_pb.dataset_name][self.row_*self.batch_size_:(self.row_+1)*self.batch_size_].reshape(-1, self.frame_size_)
    data_32 = self.df[self.data_pb.dataset_name+'_64'][self.row_*self.batch_size_:(self.row_+1)*self.batch_size_].reshape(-1, 64*64)
    target = self.df[self.data_pb.target_name][self.row_*self.batch_size_:(self.row_+1)*self.batch_size_].reshape(-1, self.frame_size_)
    self.row_ += 1
    if self.row_ == self.dataset_size_/self.batch_size_:
      self.row_ = 0
    return data, data_32, target

  def GetLabeledMRBatch(self, verbose=False):
    data = self.df[self.data_pb.dataset_name][self.row_*self.batch_size_:(self.row_+1)*self.batch_size_].reshape(-1, self.frame_size_)
    data_32 = self.df[self.data_pb.dataset_name+'_64'][self.row_*self.batch_size_:(self.row_+1)*self.batch_size_].reshape(-1, 64*64)
    target = self.df[self.data_pb.target_name][self.row_*self.batch_size_:(self.row_+1)*self.batch_size_].reshape(-1, self.frame_size_)
    labels = self.df[self.data_pb.labels_name][self.row_*self.batch_size_:(self.row_+1)*self.batch_size_]
    self.row_ += 1
    if self.row_ == self.dataset_size_/self.batch_size_:
      self.row_ = 0
    return data, data_32, target, labels

  def GetScrubMRBatch(self, verbose=False):
    data = self.df[self.data_pb.dataset_name][self.row_*self.batch_size_:(self.row_+1)*self.batch_size_].reshape(-1, self.frame_size_)
    data_32 = self.df[self.data_pb.dataset_name+'_64'][self.row_*self.batch_size_:(self.row_+1)*self.batch_size_].reshape(-1, 64*64)
    target = self.df[self.data_pb.target_name][self.row_*self.batch_size_:(self.row_+1)*self.batch_size_].reshape(-1, self.frame_size_)
    labels = self.df[self.data_pb.labels_name][self.row_*self.batch_size_:(self.row_+1)*self.batch_size_].reshape(-1, 1)
    self.row_ += 1
    if self.row_ == 62741/self.batch_size_:
      self.row_ = 0
    return data, data_32, target, labels

class LFW128DataHandler(object):
  def __init__(self, data_pb):
    self.seq_length_ = data_pb.num_frames
    self.row_length_ = data_pb.row_length
    self.col_length_ = data_pb.col_length
    self.stride_ = data_pb.stride
    self.stride_dec_ = data_pb.stride_dec
    self.batch_size_ = data_pb.batch_size
    self.image_size_ = data_pb.image_size
    self.dataset_size_ = data_pb.dataset_size # 13233
    self.frame_size_ = self.image_size_ ** 2
    f = h5py.File(data_pb.data_file)
    self.data_ = f[data_pb.dataset_name].value.reshape(-1, self.frame_size_)
    self.data_32_ = f[data_pb.dataset_name+'_64'].value.reshape(-1, 64*64)
    self.target_ = f[data_pb.target_name].value.reshape(-1, self.frame_size_)
    self.labels_ = f[data_pb.labels_name].value
    self.filenames_ = f[data_pb.files_name].value
    f.close()
    self.indices_ = np.arange(self.dataset_size_)
    self.row_ = 0
    # np.random.shuffle(self.indices_)

  def GetBatchSize(self):
    return self.batch_size_

  def GetDims(self):
    return self.frame_size_

  def GetDatasetSize(self):
    return self.dataset_size_

  def GetSeqLength(self):
    return self.seq_length_

  def GetRowLength(self):
    return self.row_length_

  def GetColLength(self):
    return self.col_length_

  def Reset(self):
    self.row_ = 0

  def GetTestBatch(self, verbose=False):
    data = np.zeros((self.batch_size_, self.frame_size_), dtype=np.float32)
    data_32 = np.zeros((self.batch_size_, 64*64), dtype=np.float32)
    target = np.zeros((self.batch_size_, self.frame_size_), dtype=np.float32)
    labels = np.empty((self.batch_size_, 1), dtype='|S40')
    names = np.empty((self.batch_size_, 1), dtype='|S45')
    for j in xrange(self.batch_size_):
      ind = self.indices_[self.row_]
      self.row_ += 1
      if self.row_ == self.dataset_size_:
        self.row_ = 0
        np.random.shuffle(self.indices_)
      data[j, :] = self.data_[ind, :]
      data_32[j, :] = self.data_32_[ind, :]
      target[j, :] = self.target_[ind, :]
      labels[j, :] = self.labels_[ind, :]
      names[j, :] = self.filenames_[ind, :]
    return data, data_32, target, labels, names

  def GetVaeTestBatch(self, verbose=False):
    data = np.zeros((self.batch_size_, 1, self.image_size_, self.image_size_), dtype=np.float32)
    target = np.zeros((self.batch_size_, 1, self.image_size_, self.image_size_), dtype=np.float32)
    for j in xrange(self.batch_size_):
      ind = self.indices_[self.row_]
      self.row_ += 1
      if self.row_ == self.dataset_size_:
        self.row_ = 0
        np.random.shuffle(self.indices_)
      data[j, :] = self.data_[ind, :].reshape(1, self.image_size_, self.image_size_)
      target[j, :] = self.target_[ind, :].reshape(1, self.image_size_, self.image_size_)
    return data, target

class OccFaceSTNDataHandler(object):
  def __init__(self, data_pb):
    self.data_pb = data_pb
    self.batch_size_ = data_pb.batch_size
    self.image_size_ = data_pb.image_size
    self.num_person = 9500
    if self.batch_size_ == 128:
      self.dataset_size_ = 2737 * self.batch_size_
      self.dataset_test_size_ = 306 * self.batch_size_
    elif self.batch_size_ == 256:
      self.dataset_size_ = 1368 * self.batch_size_
      self.dataset_test_size_ = 153 * self.batch_size_
    self.frame_size_ = self.image_size_ ** 2
    self.df = h5py.File(data_pb.data_file)
    self.indices_ = np.arange(self.dataset_size_)
    self.row_ = 0
    np.random.shuffle(self.indices_)

  def GetBatchSize(self):
    return self.batch_size_

  def GetDims(self):
    return self.frame_size_

  def GetDatasetSize(self):
    return self.dataset_size_

  def GetDatasetTestSize(self):
    return self.dataset_test_size_

  def Reset(self):
    self.row_ = 0

  def GetBatch(self, verbose=False):
    data = self.df[self.data_pb.dataset_name][self.row_*self.batch_size_:(self.row_+1)*self.batch_size_]
    target = self.df[self.data_pb.target_name][self.row_*self.batch_size_:(self.row_+1)*self.batch_size_]
    self.row_ += 1
    if self.row_ == self.dataset_size_/self.batch_size_:
      self.row_ = 0
    data = data/255.0
    target = target/255.0
    return data, target

  def GetTestBatch(self, verbose=False):
    data = self.df[self.data_pb.dataset_name][self.row_*self.batch_size_:(self.row_+1)*self.batch_size_]
    target = self.df[self.data_pb.target_name][self.row_*self.batch_size_:(self.row_+1)*self.batch_size_]
    self.row_ += 1
    if self.row_ == self.dataset_test_size_/self.batch_size_:
      self.row_ = 0
    data = data/255.0
    target = target/255.0
    return data, target

class MNISTDataHandler(object):
  def __init__(self, data_pb):
    self.seq_length_ = data_pb.num_frames
    self.batch_size_ = data_pb.batch_size
    self.image_size_ = data_pb.image_size
    if self.batch_size_ == 256:
      self.dataset_size_ = 117 * 2 * self.batch_size_  # The dataset is really infinite. This is just for validation.
    else:
      self.dataset_size_ = 117 * self.batch_size_
    self.frame_size_ = self.image_size_ ** 2
    f = h5py.File(data_pb.data_file)
    self.data_ = f[data_pb.dataset_name].value.reshape(-1, self.frame_size_)
    self.target_ = f[data_pb.target_name].value.reshape(-1, self.frame_size_)
    f.close()
    self.indices_ = np.arange(self.dataset_size_)
    self.row_ = 0
    # np.random.shuffle(self.indices_)

  def GetBatchSize(self):
    return self.batch_size_

  def GetDims(self):
    return self.frame_size_

  def GetDatasetSize(self):
    return self.dataset_size_

  def GetSeqLength(self):
    return self.seq_length_

  def Reset(self):
    self.row_ = 0

  def Overlap(self, a, b):
    """ Put b on top of a."""
    return np.maximum(a, b)
    #return b

  def GetBatch(self, verbose=False):
    data = np.zeros((self.batch_size_, self.frame_size_), dtype=np.float32)
    target = np.zeros((self.batch_size_, self.frame_size_), dtype=np.float32)
    for j in xrange(self.batch_size_):
      ind = self.indices_[self.row_]
      self.row_ += 1
      if self.row_ == self.dataset_size_:
        self.row_ = 0
        np.random.shuffle(self.indices_)
      data[j, :] = self.data_[ind, :]
      target[j, :] = self.target_[ind, :]
    return data, target

  def GetTestBatch(self, verbose=False):
    data = np.zeros((self.batch_size_, self.frame_size_), dtype=np.float32)
    target = np.zeros((self.batch_size_, self.frame_size_), dtype=np.float32)
    for j in xrange(self.batch_size_):
      # ind = self.indices_[self.row_]
      # self.row_ += 1
      # if self.row_ == self.dataset_size_:
      #   self.row_ = 0
      #   np.random.shuffle(self.indices_)
      data[j, :] = self.data_[self.dataset_size_+j, :]
      target[j, :] = self.target_[self.dataset_size_+j, :]
    return data, target

def Test():
  #load data from disk to cpu
  data_pb = ReadDataProto(sys.argv[1])
  buf = CPUSequenceBuffer(data_pb)
  batch_size = data_pb.batch_size
  dataset_size = buf.GetDataSetSize()
  num_dims = buf.GetDims()
  data_copy = np.zeros((dataset_size, num_dims))
  num_batches = dataset_size/batch_size
  left_overs = dataset_size % batch_size
  print num_batches
  for i in xrange(num_batches):
    print i
    data_copy[i * batch_size: (i+1)* batch_size,:] = buf.GetBatch(batch_size)[0, :, :]
  if left_overs > 0:
    data_copy[num_batches * batch_size:, :] = buf.GetBatch(left_overs)[0, :, :]

  # These should match.
  print data_copy[:5, :5]
  print buf.GetBatch(5)[0, :, :5]

if __name__ == '__main__':
  Test()

