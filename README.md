Convolutional VAE with MNIST. Notes for convolutional are in the very excellent theano page

http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html


Builds upon pytorch's vanilla mnist and vae examples 
https://github.com/pytorch/examples/blob/master/mnist/main.py
https://github.com/pytorch/examples/tree/master/vae/main.py

It adds convolutions to the VAE's encoder. 

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # 1 inp channel, 10 output channels
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # 
        self.conv2_drop = nn.Dropout2d() 

        
        self.fc1 = nn.Linear(320, 100)
        self.fc21 = nn.Linear(100, 20)
        self.fc22 = nn.Linear(100, 20)
        self.fc3 = nn.Linear(20, 100)
        self.fc4 = nn.Linear(100, 784)


conv_VAE.py: encoder has convolutions. Decoder is fully connected
convdeconv_VAE: encoder has convolutions. Decoder has a deconv architecture but not the same as encoder. Still experimenting.
convdeconv_stride_VAE: This one has strided convolutions/deconvolutions instead of max pooling.

