"""
这个代码用于attention及其可视化的学习地址来源:https://medium.com/datalogue/attention-in-keras-1892773a4f22
"""
def call(self, x):
    # store the whole sequence so we can "attend" to it at each timestep
    self.x_seq = x

    # apply the a dense layer over the time dimension of the sequence
    # do it here because it doesn't depend on any previous steps
    # and thefore we can save computation time:
    
    self._uxpb = _time_distributed_dense(self.x_seq, self.U_a, b=self.b_a,
                                         input_dim=self.input_dim,
                                         timesteps=self.timesteps,
                                         output_dim=self.units)

    return super(AttentionDecoder, self).call(x)
  """
  这里的x对应h表示近期记忆，用于计算e(j,t),将e(j,t)做类似于softmax的操作可以得到attention层的权重alpha
  """
  
