import keras
import numpy as np 
from keras import layers
from keras import backend as K
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
latent_dim = 2
batch_size = 16
maxlen = 30
#max_words = 28
traj_shape = (maxlen,)
input_traj = keras.Input(shape=traj_shape)
# 编码器
embedded_traj = layers.Embedding(input_dim=28,output_dim=8)(input_traj)
embeddt = Model(input_traj,embedded_traj)
print('\n------embedding model:\n')
embeddt.summary()   
encoded_traj = layers.LSTM(100,return_sequences=True)(embedded_traj)


z_mean = layers.Dense(latent_dim)(encoded_traj)
z_log_var = layers.Dense(latent_dim)(encoded_traj)

#潜在空间采样的函数
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0],latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var)*epsilon
z = layers.Lambda(sampling)([z_mean, z_log_var])


#VAE解码器网络，将潜在空间点映射为图像
decoder_input = layers.Input(K.int_shape(z)[1:])
decoded_traj = layers.LSTM(100,return_sequences=True)(decoder_input)

#将模型实例化，解码为原轨迹
decoder = Model(decoder_input, decoded_traj)
print('\n------decoder model:\n')
decoder.summary()
#将实例应用于z,得到解码后的z
z_decoded = decoder(z)


#用于计算VAE损失的自定义层
class CustomVariationalLayer(keras.layers.Layer):
    
    def vae_loss(self, x, z_decoded):
        x = embeddt(x)
        #print('\n-------x.shape:',x.shape)
        x = K.flatten(x)
        #print('\n-------x.shape:',x.shape)
        z_decoded = K.flatten(z_decoded)
        #print('\n-------z_decoded.shape:',z_decoded.shape)
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * K.mean(1 + z_log_var- K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x
    
y = CustomVariationalLayer()([input_traj, z_decoded])

vae = Model(input_traj, y)
vae.compile(optimizer='rmsprop', loss=None)
print('\n------vae model:\n')
vae.summary()

#读取数据
r_filename = 'CATHI/magickdata.txt'
f = open(r_filename)   
traj = list()
for line in f.readlines():                # 逐行读取数据
    line = line.strip()                #去掉每行头尾空白
    if not len(line) or line.startswith('#'):   # 判断是否是空行或注释行
        continue                  #是的话，跳过不处理
    line = [int(item) for item in line.split(sep=' ')]
    traj.append(line)              #保存

# print(traj[:10])


x_train = pad_sequences(traj[:-100], maxlen=maxlen)
x_test = pad_sequences(traj[-100:], maxlen=maxlen)
print(x_train.shape)
vae.fit(x=x_train, y=None, 
        shuffle=True, 
        epochs=10, 
        batch_size=batch_size, 
        validation_data=(x_test, None))
