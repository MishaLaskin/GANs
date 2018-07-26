# GANs

You can view a write-up on the subject here 

https://paper.dropbox.com/doc/GANs--AIgavHo4deGgaJ8~EP4dIv0tAQ-FawjaaJP9R7dDYlTfcvdX

# GANs
We’ll use a simple GAN to generate frog images from the CIFAR dataset


## The Dataset

CIFAR has 10 image classes with **5,000 training images** of **dimension 32X32** in RGB.


## Image Generation Program


1. A generator network $$G$$ maps images of shape `(latent_dim,)` to `(32,32,3)` , so the generator map is $$G: \mathbb{R}\rightarrow \mathbb{R}^3$$
2. A discriminator network $$D$$ maps the images from `(32,32,3)`, to a binary score $$$$`[0,1]` predicting whether the image is real or fake
3.  A GAN network chains $$D$$ and $$G$$ so $$GAN = D(G(x))$$, so the GAN map is $$G: \mathbb{R} \rightarrow \mathbb{R}^3 \rightarrow [0,1]$$, from `(latent_dim,)` to `(32,32,3)` to `[0,1]`
4. $$D$$ is trained with a collection of fake images (from $$G$$) and real images from the training set
5. To train $$G$$, you adjust the weights in $$G$$ with regard to the loss of the $$GAN$$ model, so $$G$$ weights are updated in the directions that makes $$D$$ more likely classify fake images as real



## The Generator 


    import keras
    from keras import layers
    import numpy as np
    
    #latent vector
    latent_dim = 32
    height = 32
    width = 32
    channels = 3
    
    generator_input = keras.Input(shape=(latent_dim,))
    
    #Transforms the generator input (latent_dim,) to a 16X16 128 channel feature map
    x = layers.Dense(128 * 16 * 16)(generator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((16, 16, 128))(x)
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)
    
    #Transforms to 32 X 32
    x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)
    
    # These 3 lines produce a 32X32 1 channel image, which is the shape of CIFAR10 images
    x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)\
    
    # Instantiates generator model which maps (latent_dim,) to (32,32,3)
    generator = keras.models.Model(generator_input, x)
    generator.summary()


## The Discriminator


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
    
    # Trick 1: Dropout layer
    x = layers.Dropout(0.4)(x)
    # Classification step
    x = layers.Dense(1, activation='sigmoid')(x)
    # instantiates discriminator which turns (32,32,3) to [0,1]
    discriminator = keras.models.Model(discriminator_input, x)
    discriminator.summary()
    discriminator_optimizer = keras.optimizers.RMSprop(
        lr=0.0008,
        clipvalue=1.0,
        decay=1e-8)
    discriminator.compile(optimizer=discriminator_optimizer,loss='binary_crossentropy')



## The Adversarial Network


    # This sets discriminator weights as untrainable
    discriminator.trainable = False
    
    gan_input = keras.Input(shape=(latent_dim,)) 
    
    gan_output = discriminator(generator(gan_input)) 
    
    gan = keras.models.Model(gan_input, gan_output)
    
    gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8) 
    
    gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')



## Training The GAN

GAN Algorithm Steps

**Generate Data**


1. Create random latent space data (random noise in 1D vector) - shape `(latent_dim,)`
2. Generate images from latent space date - shape `(32,32,3)`

**Mix Fake + Real Data**


3. Mix real + fake images in one dataset (with targets *real* as 1 and *fake* as 0)

**Train Discriminator**


4. Train discriminator with these images

**Train Generator to Fool Discriminator**


5. Create new latent space random vectors
6. Label them as *real* 
7. Train GAN with new random vectors (this step teaches G to fool D)


