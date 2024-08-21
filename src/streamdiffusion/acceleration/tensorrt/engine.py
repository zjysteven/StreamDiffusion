from typing import *

import torch
from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTinyOutput
from diffusers.models.controlnet import ControlNetOutput
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.autoencoders.vae import DecoderOutput
from polygraphy import cuda

from .utilities import Engine


class UNet2DConditionControlNetModelEngine:
    def __init__(
        self, 
        filepath: str, 
        stream: cuda.Stream, 
        use_cuda_graph: bool = False
    ):
        self.engine = Engine(filepath)
        self.stream = stream
        self.use_cuda_graph = use_cuda_graph

        self.engine.load()
        self.engine.activate()

    def __call__(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image,
        **kwargs,
    ) -> Any:
        if timestep.dtype != torch.float32:
            timestep = timestep.float()

        # -- DEBUG -- Print some feed dict information TODO
        #print('<< UNet2D::__call__() >>')
        #print(f'\tsample shape: {latent_model_input.shape}')
        #print(f'\ttimestep shape: {timestep.shape}')
        #print(f'\tencoder hidden states shape: {encoder_hidden_states.shape}')
        #print(f'\timage shape: {image.shape}')

        self.engine.allocate_buffers(
            shape_dict={
                "sample": latent_model_input.shape,
                "timestep": timestep.shape,
                "encoder_hidden_states": encoder_hidden_states.shape,
                "image": image.shape,
                "latent": latent_model_input.shape,
            },
            device=latent_model_input.device,
        )

        noise_pred = self.engine.infer(
            {
                "sample": latent_model_input,
                "timestep": timestep,
                "encoder_hidden_states": encoder_hidden_states,
                "image": image,
            },
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )["latent"]

        #print(f'\tnoise_pred shape: {noise_pred.shape}')

        return UNet2DConditionOutput(sample=noise_pred)

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass


class UNet2DConditionModelEngine:
    def __init__(
        self, 
        filepath: str, 
        stream: cuda.Stream, 
        use_cuda_graph: bool = False
    ):
        self.engine = Engine(filepath)
        self.stream = stream
        self.use_cuda_graph = use_cuda_graph

        self.engine.load()
        self.engine.activate()

    def __call__(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        **kwargs,
    ) -> Any:
        if timestep.dtype != torch.float32:
            timestep = timestep.float()

        self.engine.allocate_buffers(
            shape_dict={
                "sample": latent_model_input.shape,
                "timestep": timestep.shape,
                "encoder_hidden_states": encoder_hidden_states.shape,
                "latent": latent_model_input.shape,
            },
            device=latent_model_input.device,
        )

        noise_pred = self.engine.infer(
            {
                "sample": latent_model_input,
                "timestep": timestep,
                "encoder_hidden_states": encoder_hidden_states,
            },
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )["latent"]
        return UNet2DConditionOutput(sample=noise_pred)

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass


"""
Separation of ControlNet and SD U-Net into
separate engines for individual quantization control.
"""
class ControlNetModelEngine:
    def __init__(
        self,
        filepath: str,
        stream: cuda.Stream,
        use_cuda_graph: bool = False
    ):
        self.engine = Engine(filepath)
        self.stream = stream
        self.use_cuda_graph = use_cuda_graph

        # TODO: down/mid block residual sample dimensions obtained by
        #       experimentation. Hard-coded for now
        self.dbrs_list_size = 12
        self.dbrs_dim1 = 320

        self.engine.load()
        self.engine.activate()

    def __call__(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image,
        **kwargs,
    ) -> Any:
        if timestep.dtype != torch.float32:
            timestep = timestep.float()

        # -- DEBUG -- Print some feed dict information TODO
        #print('<< UNet2D::__call__() >>')
        #print(f'\tsample shape: {latent_model_input.shape}')
        #print(f'\ttimestep shape: {timestep.shape}')
        #print(f'\tencoder hidden states shape: {encoder_hidden_states.shape}')
        #print(f'\timage shape: {image.shape}')

        dim0 = latent_model_input.shape[0]
        dim2 = latent_model_input.shape[2]
        dim3 = latent_model_input.shape[3]
        dbrs_dims = [
            torch.Size([dim0, self.dbrs_dim1, dim2, dim3]),
            torch.Size([dim0, self.dbrs_dim1, dim2, dim3]),
            torch.Size([dim0, self.dbrs_dim1, dim2, dim3]),
            torch.Size([dim0, self.dbrs_dim1, dim2 // 2, dim3 // 2]),
            torch.Size([dim0, self.dbrs_dim1 * 2, dim2 // 2, dim3 // 2]),
            torch.Size([dim0, self.dbrs_dim1 * 2, dim2 // 2, dim3 // 2]),
            torch.Size([dim0, self.dbrs_dim1 * 2, dim2 // 4, dim3 // 4]),
            torch.Size([dim0, self.dbrs_dim1 * 4, dim2 // 4, dim3 // 4]),
            torch.Size([dim0, self.dbrs_dim1 * 4, dim2 // 4, dim3 // 4]),
            torch.Size([dim0, self.dbrs_dim1 * 4, dim2 // 8, dim3 // 8]),
            torch.Size([dim0, self.dbrs_dim1 * 4, dim2 // 8, dim3 // 8]),
            torch.Size([dim0, self.dbrs_dim1 * 4, dim2 // 8, dim3 // 8])
        ]
        mbrs_dim = torch.Size([dim0, int(self.dbrs_dim1 * 4), int(dim2 / 8), int(dim3 / 8)])

        self.engine.allocate_buffers(
            shape_dict={
                "sample": latent_model_input.shape,
                "timestep": timestep.shape,
                "encoder_hidden_states": encoder_hidden_states.shape,
                "image": image.shape,
                # TODO: Is there a better way to allocate this than by hard-coding every block?
                "down_block_res_samples_0": dbrs_dims[0],
                "down_block_res_samples_1": dbrs_dims[1],
                "down_block_res_samples_2": dbrs_dims[2],
                "down_block_res_samples_3": dbrs_dims[3],
                "down_block_res_samples_4": dbrs_dims[4],
                "down_block_res_samples_5": dbrs_dims[5],
                "down_block_res_samples_6": dbrs_dims[6],
                "down_block_res_samples_7": dbrs_dims[7],
                "down_block_res_samples_8": dbrs_dims[8],
                "down_block_res_samples_9": dbrs_dims[9],
                "down_block_res_samples_10": dbrs_dims[10],
                "down_block_res_samples_11": dbrs_dims[11],
                "mid_block_res_sample": mbrs_dim,
            },
            device=latent_model_input.device,
        )

        tensor_out = self.engine.infer(
            {
                "sample": latent_model_input,
                "timestep": timestep,
                "encoder_hidden_states": encoder_hidden_states,
                "image": image,
            },
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )

        return (
            tensor_out["down_block_res_samples_0"],
            tensor_out["down_block_res_samples_1"],
            tensor_out["down_block_res_samples_2"],
            tensor_out["down_block_res_samples_3"],
            tensor_out["down_block_res_samples_4"],
            tensor_out["down_block_res_samples_5"],
            tensor_out["down_block_res_samples_6"],
            tensor_out["down_block_res_samples_7"],
            tensor_out["down_block_res_samples_8"],
            tensor_out["down_block_res_samples_9"],
            tensor_out["down_block_res_samples_10"],
            tensor_out["down_block_res_samples_11"],
            tensor_out["mid_block_res_sample"]
        )

        """
        dbrs = (
            tensor_out["down_block_res_samples_0"],
            tensor_out["down_block_res_samples_1"],
            tensor_out["down_block_res_samples_2"],
            tensor_out["down_block_res_samples_3"],
            tensor_out["down_block_res_samples_4"],
            tensor_out["down_block_res_samples_5"],
            tensor_out["down_block_res_samples_6"],
            tensor_out["down_block_res_samples_7"],
            tensor_out["down_block_res_samples_8"],
            tensor_out["down_block_res_samples_9"],
            tensor_out["down_block_res_samples_10"],
            tensor_out["down_block_res_samples_11"]
        )
        mbrs = tensor_out["mid_block_res_sample"]

        return ControlNetOutput(down_block_res_samples=dbrs,
                                mid_block_res_sample=mbrs)
        """

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass


"""
Separation of ControlNet and SD U-Net into
separate engines for individual quantization control.
"""
class UNet2DNoControlNetModelEngine:
    def __init__(
        self, 
        filepath: str, 
        stream: cuda.Stream, 
        use_cuda_graph: bool = False
    ):
        self.engine = Engine(filepath)
        self.stream = stream
        self.use_cuda_graph = use_cuda_graph

        self.engine.load()
        self.engine.activate()

    def __call__(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        down_block_res_samples_0: torch.Tensor,
        down_block_res_samples_1: torch.Tensor,
        down_block_res_samples_2: torch.Tensor,
        down_block_res_samples_3: torch.Tensor,
        down_block_res_samples_4: torch.Tensor,
        down_block_res_samples_5: torch.Tensor,
        down_block_res_samples_6: torch.Tensor,
        down_block_res_samples_7: torch.Tensor,
        down_block_res_samples_8: torch.Tensor,
        down_block_res_samples_9: torch.Tensor,
        down_block_res_samples_10: torch.Tensor,
        down_block_res_samples_11: torch.Tensor,
        #down_block_res_samples,
        mid_block_res_sample: torch.Tensor,
        **kwargs,
    ) -> Any:
        if timestep.dtype != torch.float32:
            timestep = timestep.float()

        self.engine.allocate_buffers(
            shape_dict={
                "sample": latent_model_input.shape,
                "timestep": timestep.shape,
                "encoder_hidden_states": encoder_hidden_states.shape,
                "down_block_res_samples_0": down_block_res_samples_0.shape,
                "down_block_res_samples_1": down_block_res_samples_1.shape,
                "down_block_res_samples_2": down_block_res_samples_2.shape,
                "down_block_res_samples_3": down_block_res_samples_3.shape,
                "down_block_res_samples_4": down_block_res_samples_4.shape,
                "down_block_res_samples_5": down_block_res_samples_5.shape,
                "down_block_res_samples_6": down_block_res_samples_6.shape,
                "down_block_res_samples_7": down_block_res_samples_7.shape,
                "down_block_res_samples_8": down_block_res_samples_8.shape,
                "down_block_res_samples_9": down_block_res_samples_9.shape,
                "down_block_res_samples_10": down_block_res_samples_10.shape,
                "down_block_res_samples_11": down_block_res_samples_11.shape,
                "mid_block_res_sample": mid_block_res_sample.shape,
                "latent": latent_model_input.shape,
            },
            device=latent_model_input.device,
        )

        noise_pred = self.engine.infer(
            {
                "sample": latent_model_input,
                "timestep": timestep,
                "encoder_hidden_states": encoder_hidden_states,
                "down_block_res_samples_0": down_block_res_samples_0,
                "down_block_res_samples_1": down_block_res_samples_1,
                "down_block_res_samples_2": down_block_res_samples_2,
                "down_block_res_samples_3": down_block_res_samples_3,
                "down_block_res_samples_4": down_block_res_samples_4,
                "down_block_res_samples_5": down_block_res_samples_5,
                "down_block_res_samples_6": down_block_res_samples_6,
                "down_block_res_samples_7": down_block_res_samples_7,
                "down_block_res_samples_8": down_block_res_samples_8,
                "down_block_res_samples_9": down_block_res_samples_9,
                "down_block_res_samples_10": down_block_res_samples_10,
                "down_block_res_samples_11": down_block_res_samples_11,
                "mid_block_res_sample": mid_block_res_sample,
            },
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )["latent"]

        """
        self.engine.allocate_buffers(
            shape_dict={
                "sample": latent_model_input.shape,
                "timestep": timestep.shape,
                "encoder_hidden_states": encoder_hidden_states.shape,
                "down_block_res_samples_0": down_block_res_samples[0].shape,
                "down_block_res_samples_1": down_block_res_samples[1].shape,
                "down_block_res_samples_2": down_block_res_samples[2].shape,
                "down_block_res_samples_3": down_block_res_samples[3].shape,
                "down_block_res_samples_4": down_block_res_samples[4].shape,
                "down_block_res_samples_5": down_block_res_samples[5].shape,
                "down_block_res_samples_6": down_block_res_samples[6].shape,
                "down_block_res_samples_7": down_block_res_samples[7].shape,
                "down_block_res_samples_8": down_block_res_samples[8].shape,
                "down_block_res_samples_9": down_block_res_samples[9].shape,
                "down_block_res_samples_10": down_block_res_samples[10].shape,
                "down_block_res_samples_11": down_block_res_samples[11].shape,
                "mid_block_res_sample": mid_block_res_sample.shape,
                "latent": latent_model_input.shape,
            },
            device=latent_model_input.device,
        )

        noise_pred = self.engine.infer(
            {
                "sample": latent_model_input,
                "timestep": timestep,
                "encoder_hidden_states": encoder_hidden_states,
                "down_block_res_samples_0": down_block_res_samples[0],
                "down_block_res_samples_1": down_block_res_samples[1],
                "down_block_res_samples_2": down_block_res_samples[2],
                "down_block_res_samples_3": down_block_res_samples[3],
                "down_block_res_samples_4": down_block_res_samples[4],
                "down_block_res_samples_5": down_block_res_samples[5],
                "down_block_res_samples_6": down_block_res_samples[6],
                "down_block_res_samples_7": down_block_res_samples[7],
                "down_block_res_samples_8": down_block_res_samples[8],
                "down_block_res_samples_9": down_block_res_samples[9],
                "down_block_res_samples_10": down_block_res_samples[10],
                "down_block_res_samples_11": down_block_res_samples[11],
                "mid_block_res_sample": mid_block_res_sample,
            },
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )["latent"]
        """

        return UNet2DConditionOutput(sample=noise_pred)

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass


class AutoencoderKLEngine:
    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        stream: cuda.Stream,
        scaling_factor: int,
        use_cuda_graph: bool = False,
    ):
        self.encoder = Engine(encoder_path)
        self.decoder = Engine(decoder_path)
        self.stream = stream
        self.vae_scale_factor = scaling_factor
        self.use_cuda_graph = use_cuda_graph

        self.encoder.load()
        self.decoder.load()
        self.encoder.activate()
        self.decoder.activate()

    def encode(self, images: torch.Tensor, **kwargs):

        # -- DEBUG -- Print some feed dict information TODO
        #print('<< AutoencoderKLEngine::encode() >>')
        #print(f'\timages shape: {images.shape}')

        self.encoder.allocate_buffers(
            shape_dict={
                "images": images.shape,
                "latent": (
                    images.shape[0],
                    4,
                    images.shape[2] // self.vae_scale_factor,
                    images.shape[3] // self.vae_scale_factor,
                ),
            },
            device=images.device,
        )
        latents = self.encoder.infer(
            {"images": images},
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )["latent"]
        return AutoencoderTinyOutput(latents=latents)

    def decode(self, latent: torch.Tensor, **kwargs):

        # -- DEBUG -- Print some feed dict information TODO
        #print('<< AutoencoderKLEngine::decode() >>')
        #print(f'\tlatent shape: {latent.shape}')

        self.decoder.allocate_buffers(
            shape_dict={
                "latent": latent.shape,
                "images": (
                    latent.shape[0],
                    3,
                    latent.shape[2] * self.vae_scale_factor,
                    latent.shape[3] * self.vae_scale_factor,
                ),
            },
            device=latent.device,
        )
        images = self.decoder.infer(
            {"latent": latent},
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )["images"]
        return DecoderOutput(sample=images)

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass
