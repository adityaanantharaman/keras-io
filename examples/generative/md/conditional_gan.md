# Conditional GAN

**Author:** Aditya Anantharaman [twitter:ady_anr_]<br>
**Date created:** 2020/10/21<br>
**Last modified:** 2020/10/21<br>
**Description:** Conditional gan using BCE loss trained on MNIST dataset.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/conditional_gan.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/generative/conditional_gan.py)



# Training a Conditional GAN using BCE-loss on the MNIST-Dataset
Conditional gans differ from normal dc-gans in that the class of the generated images can
also be controlled. This feature is called controllable generation and is achieved by
training both the discriminator and generator with labeled data as opposed to the
unlabeled data used to train normal dc-gans. The main distinction here, is that the
discriminator in addition to predicting wheather or not the generated image is a real
looking image in general, also checks if it is a real looking image of that particular
class. How the labels are used in the training of the discriminator and generator is
discussed later in the notebook.


```python
import tensorflow as tf

tf.enable_eager_execution()
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
```

<div class="k-default-codeblock">
```
/home/aditya/.local/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:516: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint8 = np.dtype([("qint8", np.int8, 1)])
/home/aditya/.local/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:517: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
/home/aditya/.local/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:518: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint16 = np.dtype([("qint16", np.int16, 1)])
/home/aditya/.local/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:519: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
/home/aditya/.local/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:520: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint32 = np.dtype([("qint32", np.int32, 1)])
/home/aditya/.local/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:525: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  np_resource = np.dtype([("resource", np.ubyte, 1)])
/home/aditya/.local/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:541: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint8 = np.dtype([("qint8", np.int8, 1)])
/home/aditya/.local/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:542: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
/home/aditya/.local/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:543: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint16 = np.dtype([("qint16", np.int16, 1)])
/home/aditya/.local/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:544: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
/home/aditya/.local/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:545: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint32 = np.dtype([("qint32", np.int32, 1)])
/home/aditya/.local/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:550: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  np_resource = np.dtype([("resource", np.ubyte, 1)])

```
</div>
---
## loading and preprocessing the mnist dataset
Make note to include the labels as well here.


```python
(xtr, ytr), (_, _) = mnist.load_data()
xtr = np.expand_dims(xtr, -1)
xtr = xtr / 255.0  # bring the pixel values to the range 0 to 1
xtr = (xtr - 0.5) / 0.5  # bring pixel values to range -1 to 1
print(
    "shape of training images: " + str(xtr.shape), "shape of labels: " + str(ytr.shape)
)

traindata = tf.data.Dataset.from_tensor_slices((xtr, ytr)).batch(64)
```

<div class="k-default-codeblock">
```
shape of training images: (60000, 28, 28, 1) shape of labels: (60000,)

```
</div>
# How the labels play a role :
### Generator :
The generaotor takes in an input noise vector and generates an image. Here, the input
noise vector, and the desired class' one-hot encoding vector are concatenated, and this
is then fed into the generator in order to generate images of that particular class.
### Discriminator
In the discriminator on the other hand, the labels can be inserted into the training
process in multiple ways. Here, we add the label info into the channel-dimension of the
images fed into the discriminator. eg: the black and white mnist images have 1 channel.
The one-hot label vector is tiled into 28x28 shape using 'tf.tile' and concatenated with
the images along the 3rd dimension (channels). After tiling, if the label is of class 2,
then the 2nd channel alone will consist of all ones, while all the other 9 channels
consist of all zeros. This might be hard to grasp at first, but the implementation is
really simple and will help understand the concept better. Thus, a 28x28x1 image is
transformed into a 28x28x11 image in this case where the number of classes is 10.

---
## create generator architecture


```python

def get_generator():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                7 * 7 * 128, input_shape=(20,)
            ),  # noise vector of length 10 and the one-hot encoding of the class of length 10
            tf.keras.layers.Reshape((7, 7, 128)),
            tf.keras.layers.Conv2DTranspose(64, 3, 2, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(32, 3, 2, "same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(1, 4, 1, padding="same", activation="tanh"),
        ]
    )


gen = get_generator()
gen.summary()
```

<div class="k-default-codeblock">
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 6272)              131712    
_________________________________________________________________
reshape (Reshape)            (None, 7, 7, 128)         0         
_________________________________________________________________
conv2d_transpose (Conv2DTran (None, 14, 14, 64)        73792     
_________________________________________________________________
batch_normalization (BatchNo (None, 14, 14, 64)        256       
_________________________________________________________________
re_lu (ReLU)                 (None, 14, 14, 64)        0         
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 28, 28, 32)        18464     
_________________________________________________________________
batch_normalization_1 (Batch (None, 28, 28, 32)        128       
_________________________________________________________________
re_lu_1 (ReLU)               (None, 28, 28, 32)        0         
_________________________________________________________________
conv2d_transpose_2 (Conv2DTr (None, 28, 28, 1)         513       
=================================================================
Total params: 224,865
Trainable params: 224,673
Non-trainable params: 192
_________________________________________________________________

```
</div>
---
## create discriminator architecture


```python

def get_discriminator():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                16, 3, 2, padding="same", input_shape=(28, 28, 11)
            ),  # label info concatenated with the image
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(32, 3, 2, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(64, 3, 2, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1),
        ]
    )


dis = get_discriminator()
dis.summary()
```

<div class="k-default-codeblock">
```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 14, 14, 16)        1600      
_________________________________________________________________
batch_normalization_2 (Batch (None, 14, 14, 16)        64        
_________________________________________________________________
leaky_re_lu (LeakyReLU)      (None, 14, 14, 16)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 7, 7, 32)          4640      
_________________________________________________________________
batch_normalization_3 (Batch (None, 7, 7, 32)          128       
_________________________________________________________________
leaky_re_lu_1 (LeakyReLU)    (None, 7, 7, 32)          0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 4, 4, 64)          18496     
_________________________________________________________________
batch_normalization_4 (Batch (None, 4, 4, 64)          256       
_________________________________________________________________
leaky_re_lu_2 (LeakyReLU)    (None, 4, 4, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 1024)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 1025      
=================================================================
Total params: 26,209
Trainable params: 25,985
Non-trainable params: 224
_________________________________________________________________

```
</div>
---
## function for concatination of tensors about an axis


```python

def combine_vectors(x, y, a):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    return tf.concat([x, y], a)

```

---
## get one hot encoding


```python

def getoh(x):
    return tf.one_hot(x, 10)

```

---
## setup optimizer and loss


```python
genopt = tf.keras.optimizers.Adam(0.0002, 0.5, 0.999)
disopt = tf.keras.optimizers.Adam(0.0002, 0.5, 0.999)
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
```

---
## print grid of 25 images


```python

def showout(x):
    x = (x + 1) / 2.0
    # x = np.array(x, dtype=np.uint8)
    # x = (x).astype(np.uint8)
    fig = plt.figure(figsize=(5, 5))
    p = 0
    for i in range(x.shape[0]):
        fig.add_subplot(5, 5, p + 1)
        plt.imshow(x[i][:, :, 0].numpy().astype(np.uint8), cmap="gray")
        p += 1
    plt.show()


generated = gen(tf.random.normal((25, 20)))
print(generated.shape)
showout(generated)
```

<div class="k-default-codeblock">
```
(25, 28, 28, 1)

```
</div>
    
![png](/img/examples/generative/conditional_gan/conditional_gan_17_1.png)
    


---
## training loop


```python
genlosses = []
dislosses = []
iteration = 0
for e in range(10):
    for realdata in tqdm(traindata):
        realx = realdata[0]  # real images
        labels = realdata[1]  # labels of real images
        labels_oh = getoh(labels)
        # get one hot encoding of labels for input to generator
        curbatchsize = len(realx)
        label_channels = tf.tile(
            np.reshape(labels_oh, (curbatchsize, 1, 1, 10)), [1, 28, 28, 1]
        )
        # create tiled one-hot channels for discriminator input
        # one-hot labels are first reshaped from (batch-size, 10) into (batch-size, 1, 1, 10)
        # then this is tiled along the 2nd and 3rd dimensions to the size 28x28

        with tf.GradientTape() as distape:  # updating discriminator
            noise1 = tf.random.normal((curbatchsize, 10))
            inputvec1 = combine_vectors(noise1, labels_oh, 1)

            fakeimages1 = gen(inputvec1)
            realimages1 = realx

            f1 = combine_vectors(fakeimages1, label_channels, -1)
            # concatenate label_channels to fake images and real images
            r1 = combine_vectors(realimages1, label_channels, -1)

            fouts = dis(f1)
            routs = dis(r1)

            floss = bce(tf.zeros_like(fouts), fouts)
            rloss = bce(tf.ones_like(routs), routs)

            disloss = (floss + rloss) / 2

            disgrads = distape.gradient(disloss, dis.trainable_variables)

            disopt.apply_gradients(zip(disgrads, dis.trainable_variables))

        with tf.GradientTape() as gentape:  # updating generator
            noise2 = tf.random.normal((curbatchsize, 10))
            inputvec2 = combine_vectors(noise2, labels_oh, 1)
            fakeimages2 = gen(inputvec2)
            f2 = combine_vectors(fakeimages2, label_channels, -1)
            fouts2 = dis(f2)
            genloss = bce(tf.ones_like(fouts2), fouts2)
            gengrads = gentape.gradient(genloss, gen.trainable_variables)
            genopt.apply_gradients(zip(gengrads, gen.trainable_variables))

        genlosses.append(genloss)
        dislosses.append(disloss)
        iteration += 1
        if iteration % 500 == 0:
            # every 100 iterations, generate using a custom label-vector = CONTROLABLE GENERATION
            consta = tf.constant(
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    0,
                    1,
                    2,
                    3,
                    4,
                ]
            )
            noise3 = tf.random.normal((25, 10))
            oh = getoh(consta)
            inputvec3 = combine_vectors(noise3, oh, 1)
            fakeimages3 = gen(inputvec3)
            showout(fakeimages3)
            plt.plot(dislosses)
            plt.plot(genlosses)
            plt.show()
```


<div class="k-default-codeblock">
```
HBox(children=(HTML(value=''), FloatProgress(value=1.0, bar_style='info', layout=Layout(width='20px'), max=1.0…

WARNING:tensorflow:From /home/aditya/.local/lib/python3.6/site-packages/tensorflow/python/ops/nn_impl.py:180: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where

```
</div>
    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_2.png)
    



    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_3.png)
    


    



<div class="k-default-codeblock">
```
HBox(children=(HTML(value=''), FloatProgress(value=1.0, bar_style='info', layout=Layout(width='20px'), max=1.0…

```
</div>
    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_6.png)
    



    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_7.png)
    



    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_8.png)
    



    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_9.png)
    


    



<div class="k-default-codeblock">
```
HBox(children=(HTML(value=''), FloatProgress(value=1.0, bar_style='info', layout=Layout(width='20px'), max=1.0…

```
</div>
    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_12.png)
    



    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_13.png)
    



    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_14.png)
    



    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_15.png)
    


    



<div class="k-default-codeblock">
```
HBox(children=(HTML(value=''), FloatProgress(value=1.0, bar_style='info', layout=Layout(width='20px'), max=1.0…

```
</div>
    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_18.png)
    



    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_19.png)
    



    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_20.png)
    



    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_21.png)
    


    



<div class="k-default-codeblock">
```
HBox(children=(HTML(value=''), FloatProgress(value=1.0, bar_style='info', layout=Layout(width='20px'), max=1.0…

```
</div>
    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_24.png)
    



    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_25.png)
    



    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_26.png)
    



    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_27.png)
    


    



<div class="k-default-codeblock">
```
HBox(children=(HTML(value=''), FloatProgress(value=1.0, bar_style='info', layout=Layout(width='20px'), max=1.0…

```
</div>
    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_30.png)
    



    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_31.png)
    



    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_32.png)
    



    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_33.png)
    


    



<div class="k-default-codeblock">
```
HBox(children=(HTML(value=''), FloatProgress(value=1.0, bar_style='info', layout=Layout(width='20px'), max=1.0…

```
</div>
    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_36.png)
    



    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_37.png)
    



    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_38.png)
    



    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_39.png)
    


    



<div class="k-default-codeblock">
```
HBox(children=(HTML(value=''), FloatProgress(value=1.0, bar_style='info', layout=Layout(width='20px'), max=1.0…

```
</div>
    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_42.png)
    



    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_43.png)
    



    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_44.png)
    



    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_45.png)
    


    



<div class="k-default-codeblock">
```
HBox(children=(HTML(value=''), FloatProgress(value=1.0, bar_style='info', layout=Layout(width='20px'), max=1.0…

```
</div>
    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_48.png)
    



    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_49.png)
    


    



<div class="k-default-codeblock">
```
HBox(children=(HTML(value=''), FloatProgress(value=1.0, bar_style='info', layout=Layout(width='20px'), max=1.0…

```
</div>
    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_52.png)
    



    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_53.png)
    



    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_54.png)
    



    
![png](/img/examples/generative/conditional_gan/conditional_gan_19_55.png)
    


    


# Generating a GIF
of transitioning between the different classes by manipulation of the latent vector.
generate images from intermediate latent representations and convert the images into a
gif.


```python
consta = tf.constant(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4]
)
noise3 = tf.random.normal((25, 10))
oh = getoh(consta)
inputvec3 = combine_vectors(noise3, oh, 1)
fakeimages3 = gen(inputvec3)
showout(fakeimages3)

aa = inputvec3.numpy()
aa.shape

bb = []
for i in range(1, len(aa)):
    start = aa[i - 1]
    end = aa[i]
    inc = (end - start) / 8
    for i in range(8):
        bb.append(start + i * inc)
bb = np.array(bb)
print(bb.shape)

generated_images = gen(bb)
generated_images.shape

showout(generated_images[:25])

import imageio

imageio.mimsave("./movie.gif", generated_images.numpy())
```


    
![png](/img/examples/generative/conditional_gan/conditional_gan_21_0.png)
    


<div class="k-default-codeblock">
```
(192, 20)

```
</div>
    
![png](/img/examples/generative/conditional_gan/conditional_gan_21_2.png)
    


<div class="k-default-codeblock">
```
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999814033508301]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9998652338981628]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9978642463684082]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9987350106239319]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9994986057281494]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.999563992023468]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9991722702980042]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9991491436958313]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9997817873954773]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.999592125415802]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9994140863418579]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9992021322250366]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9988687634468079]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9987016320228577]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9986603856086731]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9993718266487122]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9996438026428223]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9996150135993958]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9996801018714905]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9996835589408875]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9996209144592285]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9995096921920776]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9995009899139404]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.999628484249115]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9997732043266296]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9998882412910461]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9996733665466309]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9987940788269043]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9991755485534668]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9997392892837524]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9999024868011475]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999517798423767]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9999746680259705]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9999803304672241]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9999779462814331]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9999620318412781]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9999744892120361]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999767541885376]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9999724626541138]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9999644756317139]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9999765157699585]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999449253082275]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9998704791069031]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9997214674949646]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9994421601295471]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.999189555644989]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9990537166595459]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.999152660369873]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9993540644645691]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9992709755897522]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9992451667785645]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.999220073223114]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9997471570968628]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9999784827232361]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999145865440369]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999868273735046]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999992251396179]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9999983310699463]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9999891519546509]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999669790267944]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999097585678101]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999272227287292]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999706149101257]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9999798536300659]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999847412109375]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9999380707740784]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9996986985206604]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9997539520263672]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9997535347938538]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9997472763061523]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9996966123580933]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.999896228313446]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999642968177795]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9998822212219238]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9997758269309998]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9997579455375671]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9997151494026184]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9996412992477417]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9996850490570068]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9998008012771606]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999274611473083]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9997953176498413]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9992976188659668]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9996203184127808]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9998473525047302]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9999340772628784]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9997859597206116]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9999294281005859]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999473094940186]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999276399612427]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9998952150344849]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9996577501296997]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9998137950897217]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9998895525932312]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9998171925544739]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.999686598777771]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9998093247413635]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9998651742935181]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9999138116836548]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999444484710693]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9999614953994751]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.99997478723526]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.999984860420227]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999908208847046]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.999993085861206]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999817609786987]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999186396598816]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9998071789741516]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9999424815177917]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999898672103882]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9999980330467224]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9999993443489075]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999997019767761]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999991059303284]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999961853027344]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999747276306152]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9998539090156555]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9997039437294006]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9996793866157532]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9998512864112854]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999472498893738]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9999477863311768]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9998642802238464]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9998477697372437]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9996861219406128]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9996326565742493]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9994598031044006]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.999622106552124]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.999596357345581]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9991704225540161]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9990369081497192]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9992877244949341]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9990916848182678]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9992715716362]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9993152618408203]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9990286231040955]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9990172386169434]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.999349057674408]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9997536540031433]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9993047714233398]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9991268515586853]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9997566938400269]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9991703033447266]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9996012449264526]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9997624158859253]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9995684027671814]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9983722567558289]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9974007606506348]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9987037181854248]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9994868636131287]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9997422695159912]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9998430609703064]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9998953938484192]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9998824596405029]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9997543096542358]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9992985129356384]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9991711974143982]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9989770650863647]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9989748597145081]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9985512495040894]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9987884759902954]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.998986005783081]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9995935559272766]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9990856051445007]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9995855689048767]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9998188018798828]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9999143481254578]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9999462366104126]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999876022338867]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999819397926331]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999641180038452]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9999147057533264]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9997975826263428]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9997233152389526]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9995198249816895]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9992671608924866]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9996694326400757]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9998989701271057]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.999955415725708]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999723434448242]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999629855155945]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999749064445496]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9999741911888123]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9999744296073914]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9999862313270569]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9999642968177795]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.999874472618103]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9998916983604431]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9999327063560486]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9999151229858398]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, 0.9998430013656616]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0000001192092896, 0.9998124241828918]. Convert image to uint8 prior to saving to suppress this warning.

```
</div>