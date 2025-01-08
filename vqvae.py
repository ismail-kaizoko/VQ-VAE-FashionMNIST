from torch import nn
import torch
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss

from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor
Tensor = TypeVar('torch.tensor')



###### Hyper Parameters of the Model ######
in_channels = 1





class VectorQuantizer(nn.Module):
    """
    This is the quantizer Block inspired by the git repository : 
    https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py
    https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25,
                 embedding: Tensor = None):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        if ( embedding == None ) : 
            self.embedding = nn.Embedding(self.K, self.D)
            self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)
        else : 
            self.embedding = nn.Embedding.from_pretrained(embedding)

    def forward(self, latents: Tensor) -> Tensor:
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), embedding_loss, self.beta * commitment_loss  # [B x D x H x W]



    def quantized_latents_hist(self, latents: Tensor) -> Tensor:
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]


        # Calculate the histogram of used embeddings
        encoding_inds_flat = encoding_inds.view(-1)  # Flatten to [BHW]
        embedding_histogram = torch.bincount(encoding_inds_flat, minlength=self.K)  # Count occurrences of each embedding
        
        return embedding_histogram



class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      nn.Conv2d(out_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)




class VQVAE(nn.Module):

    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 num_embeddings: int,
                #hidden_dims: List = None,
                 downsampling_factor :int = 2,
                 beta: float = 0.25,
                 embedding: Tensor = None,
                 **kwargs) -> None:
        super(VQVAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        modules = []
        
        if downsampling_factor < 2 :
            raise Warning("VQVAE can't have a donwsampling factor less than 2")
        elif downsampling_factor ==2 :
            hidden_dims = [32]
        elif downsampling_factor == 4 :
            hidden_dims = [32, 64]
        elif downsampling_factor == 8 :
            hidden_dims = [32, 64, 128]
        else:
            assert("downsamplig factor must be one of the following numbers : {2, 4, 8 }")



        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU())
        )

        for _ in range(1):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim,
                          kernel_size=1, stride=1),
                nn.LeakyReLU())
        )

        self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer(num_embeddings = num_embeddings,
                                        embedding_dim = embedding_dim,
                                        beta = self.beta,
                                        embedding = embedding)

        # Build Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim,
                          hidden_dims[-1],
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.LeakyReLU())
        )

        for _ in range(1):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

        modules.append(nn.LeakyReLU())

        hidden_dims.reverse()

        # for i in range(len(hidden_dims) - 1):
        #     modules.append(
        #         nn.Sequential(
        #             nn.ConvTranspose2d(hidden_dims[i],
        #                                hidden_dims[i + 1],
        #                                kernel_size=4,
        #                                stride=2,
        #                                padding=1),
        #             nn.LeakyReLU())
        #     )

        # modules.append(
        #     nn.Sequential(
        #         nn.ConvTranspose2d(hidden_dims[-1],
        #                            out_channels=1,
        #                            kernel_size=4,
        #                            stride=2, padding=1),
        #         nn.ReLU()
        #         ))

        # self.decoder = nn.Sequential(*modules)

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    # Upsample using bilinear interpolation
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    # Apply convolution to adjust channels
                    nn.Conv2d(hidden_dims[i],
                              hidden_dims[i + 1],
                              kernel_size=3,
                              padding=1),
                    nn.LeakyReLU())
            )
        
        # Final layer
        modules.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(hidden_dims[-1],
                          out_channels=1,
                          kernel_size=3,
                          padding=1),
                nn.ReLU())
        )
        
        self.decoder = nn.Sequential(*modules)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        return [result]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder(z)
        return result

    def forward(self, inputs: Tensor, **kwargs) -> List[Tensor]:
        encoding = self.encode(inputs)[0]
        quantized_inputs, embedding_loss, commitment_loss_beta = self.vq_layer(encoding)
        return [self.decode(quantized_inputs), inputs, embedding_loss, commitment_loss_beta]


    def codebook_usage(self, inputs: Tensor, **kwargs) -> List[Tensor]:
        encoding = self.encode(inputs)[0]
        quantized_hist = self.vq_layer.quantized_latents_hist(encoding)
        return quantized_hist



    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        inputs = args[1]
        embedding_loss = args[2]
        commitment_loss_beta = args[3]

        recons_loss = F.mse_loss(recons,inputs)

        loss = recons_loss + embedding_loss + commitment_loss_beta
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'CodeBook Loss':embedding_loss,
                'Embedding Loss':commitment_loss_beta}

    # def sample(self,
    #            num_samples: int,
    #            current_device: Union[int, str], **kwargs) -> Tensor:
    #     raise Warning('VQVAE sampler is not implemented.')

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return (self.forward(x)[0] > 0.5 ) # Since we are dealing with binary image.




