import gc
import os

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)
from polygraphy import cuda

from ...pipeline import StreamDiffusion
from .builder import EngineBuilder, create_onnx_path
from .engine import AutoencoderKLEngine, UNet2DConditionModelEngine, UNet2DConditionControlNetModelEngine
from .models import VAE, BaseModel, UNet, UNetControlNet, VAEEncoder


class TorchVAEEncoder(torch.nn.Module):
    def __init__(self, vae: AutoencoderKL):
        super().__init__()
        self.vae = vae

    def forward(self, x: torch.Tensor):
        # ---- DEBUG ---- TODO
        #print(f'<< TorchVAEEncoder::forward() >> Calling self.vae.encode()')
        return retrieve_latents(self.vae.encode(x))


def compile_vae_encoder(
    vae: TorchVAEEncoder,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    engine_build_options: dict = {},
    int8_calib_loader = None,

):
    builder = EngineBuilder(model_data, vae, device=torch.device("cuda"))

    if int8_calib_loader is not None:
        engine_build_options['quant_int8'] = True
        engine_build_options['quant_int8_calib_loader'] = int8_calib_loader
        #engine_build_options['quant_int8_calib_cache'] = 'vae_enc_calibration.cache'

    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        **engine_build_options,
    )


def compile_vae_decoder(
    vae: AutoencoderKL,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    engine_build_options: dict = {},
    int8_calib_loader = None,
):
    vae = vae.to(torch.device("cuda"))
    builder = EngineBuilder(model_data, vae, device=torch.device("cuda"))

    if int8_calib_loader is not None:
        engine_build_options['quant_int8'] = True
        engine_build_options['quant_int8_calib_loader'] = int8_calib_loader
        #engine_build_options['quant_int8_calib_cache'] = 'vae_dec_calibration.cache'

    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        **engine_build_options,
    )


def compile_unet(
    unet: UNet2DConditionModel,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    engine_build_options: dict = {},
    int8_calib_loader = None,
):
    unet = unet.to(torch.device("cuda"), dtype=torch.float16)
    builder = EngineBuilder(model_data, unet, device=torch.device("cuda"))

    if int8_calib_loader is not None:
        print('<< compile_unet() >> Setting build options for INT8 quantization.')
        engine_build_options['quant_int8'] = True
        engine_build_options['quant_int8_calib_loader'] = int8_calib_loader
        #engine_build_options['quant_int8_calib_cache'] = 'unet_calibration.cache'

    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        **engine_build_options,
    )


def accelerate_with_tensorrt(
    stream: StreamDiffusion,
    engine_dir: str,
    max_batch_size: int = 2,
    min_batch_size: int = 1,
    use_cuda_graph: bool = False,
    engine_build_options: dict = {},
):
    if "opt_batch_size" not in engine_build_options or engine_build_options["opt_batch_size"] is None:
        engine_build_options["opt_batch_size"] = max_batch_size
    text_encoder = stream.text_encoder
    unet = stream.unet
    vae = stream.vae

    del stream.unet, stream.vae, stream.pipe.unet, stream.pipe.vae

    vae_config = vae.config
    vae_dtype = vae.dtype

    unet.to(torch.device("cpu"))
    vae.to(torch.device("cpu"))

    gc.collect()
    torch.cuda.empty_cache()

    onnx_dir = os.path.join(engine_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    unet_engine_path = f"{engine_dir}/unet.engine"
    vae_encoder_engine_path = f"{engine_dir}/vae_encoder.engine"
    vae_decoder_engine_path = f"{engine_dir}/vae_decoder.engine"

    unet_model = UNet(
        fp16=True,
        device=stream.device,
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
        embedding_dim=text_encoder.config.hidden_size,
        unet_dim=unet.config.in_channels,
    )
    vae_decoder_model = VAE(
        device=stream.device,
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
    )
    vae_encoder_model = VAEEncoder(
        device=stream.device,
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
    )

    if not os.path.exists(unet_engine_path):
        compile_unet(
            unet,
            unet_model,
            create_onnx_path("unet", onnx_dir, opt=False),
            create_onnx_path("unet", onnx_dir, opt=True),
            unet_engine_path,
            engine_build_options,
        )
    else:
        del unet

    if not os.path.exists(vae_decoder_engine_path):
        vae.forward = vae.decode
        compile_vae_decoder(
            vae,
            vae_decoder_model,
            create_onnx_path("vae_decoder", onnx_dir, opt=False),
            create_onnx_path("vae_decoder", onnx_dir, opt=True),
            vae_decoder_engine_path,
            engine_build_options,
        )

    if not os.path.exists(vae_encoder_engine_path):
        vae_encoder = TorchVAEEncoder(vae).to(torch.device("cuda"))
        compile_vae_encoder(
            vae_encoder,
            vae_encoder_model,
            create_onnx_path("vae_encoder", onnx_dir, opt=False),
            create_onnx_path("vae_encoder", onnx_dir, opt=True),
            vae_encoder_engine_path,
            engine_build_options,
        )

    del vae

    cuda_steram = cuda.Stream()

    stream.unet = UNet2DConditionModelEngine(unet_engine_path, cuda_steram, use_cuda_graph=use_cuda_graph)
    stream.vae = AutoencoderKLEngine(
        vae_encoder_engine_path,
        vae_decoder_engine_path,
        cuda_steram,
        stream.pipe.vae_scale_factor,
        use_cuda_graph=use_cuda_graph,
    )
    setattr(stream.vae, "config", vae_config)
    setattr(stream.vae, "dtype", vae_dtype)

    gc.collect()
    torch.cuda.empty_cache()

    return stream


def accelerate_with_tensorrt_unetcontrol(
    stream: StreamDiffusion,
    engine_dir: str,
    max_batch_size: int = 2,
    min_batch_size: int = 1,
    use_cuda_graph: bool = False,
    engine_build_options: dict = {},
    #engine_dir_quant: str = '',
    quant_encoder = False,
    quant_unet = False,
    quant_decoder = False,
    encoder_calibration_loader = None,
    unet_calibration_loader = None,
    decoder_calibration_loader = None,
):
    if "opt_batch_size" not in engine_build_options or engine_build_options["opt_batch_size"] is None:
        engine_build_options["opt_batch_size"] = max_batch_size
    text_encoder = stream.text_encoder
    unet = stream.unet
    vae = stream.vae

    del stream.unet, stream.vae, stream.pipe.unet, stream.pipe.vae

    vae_config = vae.config
    vae_dtype = vae.dtype

    unet.to(torch.device("cpu"))
    vae.to(torch.device("cpu"))

    gc.collect()
    torch.cuda.empty_cache()

    # ONNX directory for both quantized and standard TRT engines
    onnx_dir = os.path.join(engine_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    # Path and filenames based on quantization flags
    unet_base = "unet" if quant_unet == False else "unet_quantINT8"
    vae_encoder_base = "vae_encoder" if quant_encoder == False else "vae_encoder_quantINT8"
    vae_decoder_base = "vae_decoder" if quant_decoder == False else "vae_decoder_quantINT8"

    unet_engine_path = f"{engine_dir}/{unet_base}.engine"
    vae_encoder_engine_path = f"{engine_dir}/{vae_encoder_base}.engine"
    vae_decoder_engine_path = f"{engine_dir}/{vae_decoder_base}.engine"

    unet_model = UNetControlNet(
        fp16=True,
        device=stream.device,
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
        embedding_dim=text_encoder.config.hidden_size,
        unet_dim=unet.unet.config.in_channels,
    )
    vae_decoder_model = VAE(
        device=stream.device,
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
    )
    vae_encoder_model = VAEEncoder(
        device=stream.device,
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
    )

    if not os.path.exists(unet_engine_path):
        compile_unet(
            unet,
            unet_model,
            create_onnx_path(unet_base, onnx_dir, opt=False),
            create_onnx_path(unet_base, onnx_dir, opt=True),
            unet_engine_path,
            engine_build_options=engine_build_options,
            int8_calib_loader=unet_calibration_loader
        )
    else:
        del unet

    if not os.path.exists(vae_decoder_engine_path):
        vae.forward = vae.decode
        compile_vae_decoder(
            vae,
            vae_decoder_model,
            create_onnx_path(vae_decoder_base, onnx_dir, opt=False),
            create_onnx_path(vae_decoder_base, onnx_dir, opt=True),
            vae_decoder_engine_path,
            engine_build_options=engine_build_options,
            int8_calib_loader=decoder_calibration_loader
        )

    if not os.path.exists(vae_encoder_engine_path):
        vae_encoder = TorchVAEEncoder(vae).to(torch.device("cuda"))
        compile_vae_encoder(
            vae_encoder,
            vae_encoder_model,
            create_onnx_path(vae_encoder_base, onnx_dir, opt=False),
            create_onnx_path(vae_encoder_base, onnx_dir, opt=True),
            vae_encoder_engine_path,
            engine_build_options=engine_build_options,
            int8_calib_loader=encoder_calibration_loader
        )

    del vae

    cuda_steram = cuda.Stream()

    stream.unet = UNet2DConditionControlNetModelEngine(unet_engine_path, cuda_steram, use_cuda_graph=use_cuda_graph)
    stream.vae = AutoencoderKLEngine(
        vae_encoder_engine_path,
        vae_decoder_engine_path,
        cuda_steram,
        stream.pipe.vae_scale_factor,
        use_cuda_graph=use_cuda_graph,
    )
    setattr(stream.vae, "config", vae_config)
    setattr(stream.vae, "dtype", vae_dtype)

    gc.collect()
    torch.cuda.empty_cache()

    return stream