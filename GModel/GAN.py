#GAN生成网络
import keras
from keras import layers
import numpy as np

latent_dim = 32
height = 32
width = 32
channels = 3

generator_input = keras.Input(shape=(latent_dim,))
#将输入转换为大小为16*16的128通道的特征图
x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16, 16, 128))(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
#上采样为32*32
x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
#将生成器模型实例化，形状为(latent_dim,),输入为(32,32,3)图像
generator = keras.models.Model(generator_input, x)
generator.summary()

#GAN判别器
discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)

x = layers.Dropout(0.4)(x)
#分类层
x = layers.Dense(1, activation='sigmoid')(x)
#将判别器模型实例化，它将输入形状为（32，32，3）的输入转化为而精致分类器决策（真，假）
discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary()
#在优化器中使用梯度裁减（限制梯度的范围）；为了稳定训练过程，使用学习率衰减
discriminator_optimizer = keras.optimizers.RMSprop(
    lr=0.0008, 
    clipvalue=1.0,
    decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')


#对抗网络
#将判别器权重设置为不可训练(仅应用于gan模型)
discriminator.trainable = False

gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)
gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')


import os
from keras.preprocessing import image

(x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()


#选择青蛙图像（类别编号为6）
x_train = x_train[y_train.flatten() == 6]
#数据标准化
x_train = x_train.reshape((x_train.shape[0],) + (height, width, channels)).astype('float32')/255.

iterations = 10000
batch_size = 20
save_dir = 'your_dir'#指定保存图像的目录

start = 0
for step in range(iterations):
    print("step:", step,'...')
    #在潜在空间采样随机点
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    #将这些点解码为虚假图像
    generated_images = generator.predict(random_latent_vectors)
    #将虚假图像和真实图像合在一起
    stop = start + batch_size
    real_images = x_train[start : stop]
    combined_images = np.concatenate([generated_images, real_images])
    #合并标签
    labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
    #向标签中添加随机噪声（重要技巧）
    labels += 0.05 * np.random.random(labels.shape)
    #训练判别器
    d_loss = discriminator.train_on_batch(combined_images, labels)
    #在潜在空间采样随机点
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    #合并标签，全部是“真实图像”（这是在撒谎）
    misleading_targets = np.zeros((batch_size, 1))
    #通过gan来训练生成器（冻结判别器的权重）
    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)
    
    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0
    if step % 100 == 0:#100步保存并绘图
        gan.save_weights('gan.h5')#保存权重
        
        print('discriminator loss:', d_loss)
        print('adversarial loss:', a_loss)
        #保存生成图像和真实图像
        img = image.array_to_img(generated_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir,'generated_frog' + str(step) + '.png'))
        img = image.array_to_img(real_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir,'real_frog' + str(step) + '.png'))