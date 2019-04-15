这个代码用于attention及其可视化的学习地址来源:https://medium.com/datalogue/attention-in-keras-1892773a4f22
```
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
```
 这里的x对应h表示近期记忆，用于计算e(j,t),将e(j,t)做类似于softmax的操作可以得到attention层的权重alpha
 下面对attention layer的核心代码进行解析。
 ```
 def step(self, x, states):
    
    # obtain elements of the previous time step.
    ytm, stm = states

    ##    ##    ##    ##    ##    ##    ##    ##    ##   
    
    # equation 1
    
    # > repeat the hidden state to the length of the sequence
    _stm = K.repeat(stm, self.timesteps)

    # > now multiplty the weight matrix with the 
    #   repeated hidden state
    _Wxstm = K.dot(_stm, self.W_a)
    
    # > calculate the unnormalized probabilities
    et = K.dot(activations.tanh(_Wxstm + self._uxpb),
               K.expand_dims(self.V_a))

    ##    ##    ##    ##    ##    ##    ##    ##    ##
    
    # equation 2 
    at = K.exp(et)
    at_sum = K.sum(at, axis=1)
    at_sum_repeated = K.repeat(at_sum, self.timesteps)
    # vector of size (batchsize, timesteps, 1)
    at /= at_sum_repeated  

    ##    ##    ##    ##    ##    ##    ##    ##    ##    
    
    # equation 3
    context = K.squeeze(
                K.batch_dot(at,
                            self.x_seq,
                            axes=1),
                axis=1)
    
    # ~~~> calculate new hidden state
    
    # equation 4  (reset gate)
    rt = activations.sigmoid(
        K.dot(ytm, self.W_r)
        + K.dot(stm, self.U_r)
        + K.dot(context, self.C_r)
        + self.b_r)

    # equation 5 (update gate)
    zt = activations.sigmoid(
        K.dot(ytm, self.W_z)
        + K.dot(stm, self.U_z)
        + K.dot(context, self.C_z)
        + self.b_z)

    # equation 6 (proposal state)
    s_tp = activations.tanh(
        K.dot(ytm, self.W_p)
        + K.dot((rt * stm), self.U_p)
        + K.dot(context, self.C_p)
        + self.b_p)

    # equation 7 (new hidden states)
    st = (1-zt)*stm + zt * s_tp
    
    # equation 8 
    # the probability of having each character.
    yt = activations.softmax(
        K.dot(ytm, self.W_o)
        + K.dot(st, self.U_o)
        + K.dot(context, self.C_o)
        + self.b_o)

    # a switch so that we can return the 
    # attention for visualizations
    if self.return_probabilities:
        return at, [yt, st]
    else:
        return yt, [yt, st]
```
这里的ytm代表的是序列前一个字母(元素) 的输出(本层输出为y(t)，ytm指代y(t-1))，stm代表的是长期记忆(s(t-1))
由于这里短期记忆输入为h(1)--->h(t)的所有项，但y(t-1)始终保持不变，因此让y(t-1)在时间节点(1--->t)上复制t次，
```
# obtain elements of the previous time step.
    ytm, stm = states

    ##    ##    ##    ##    ##    ##    ##    ##    ##   
    
    # equation 1
    
    # > repeat the hidden state to the length of the sequence
    _stm = K.repeat(stm, self.timesteps)

    # > now multiplty the weight matrix with the 
    #   repeated hidden state
    _Wxstm = K.dot(_stm, self.W_a)
    
    # > calculate the unnormalized probabilities
    et = K.dot(activations.tanh(_Wxstm + self._uxpb),
               K.expand_dims(self.V_a))
```
这里是对attention层“注意力”的计算。
```
 # equation 2 
    at = K.exp(et)
    at_sum = K.sum(at, axis=1)
    at_sum_repeated = K.repeat(at_sum, self.timesteps)
    # vector of size (batchsize, timesteps, 1)
    at /= at_sum_repeated  
```
