import json
import os
import threading
from io import BytesIO

# ! Warning: Do not remove this import, even if it's unused.
from modules.paths import script_path

from flask import Flask, request, send_file, Response

from rich import print

from modules import devices, sd_samplers
import modules.codeformer_model as codeformer
import modules.extras
import modules.face_restoration
import modules.gfpgan_model as gfpgan
import modules.img2img

import modules.lowvram
import modules.paths
import modules.scripts
import modules.sd_hijack
import modules.sd_models
import modules.shared as shared
import modules.txt2img

import modules.ui
from modules import devices
from modules import modelloader
from modules.paths import script_path
from modules.shared import cmd_opts
import modules.hypernetworks.hypernetwork

# ^ ================================================================================================================================
# ^ Helpers / Init =================================================================================================================
queue_lock = threading.Lock()

def wrap_queued_call(func):
    def f(*args, **kwargs):
        with queue_lock:
            res = func(*args, **kwargs)
        
        return res
    
    return f

def wrap_gradio_gpu_call(func, extra_outputs=None):
    def f(*args, **kwargs):
        devices.torch_gc()
        
        shared.state.sampling_step = 0
        shared.state.job_count = -1
        shared.state.job_no = 0
        shared.state.job_timestamp = shared.state.get_job_timestamp()
        shared.state.current_latent = None
        shared.state.current_image = None
        shared.state.current_image_sampling_step = 0
        shared.state.skipped = False
        shared.state.interrupted = False
        shared.state.textinfo = None
        
        with queue_lock:
            res = func(*args, **kwargs)
        
        shared.state.job = ""
        shared.state.job_count = 0
        
        devices.torch_gc()
        
        return res
    
    return modules.ui.wrap_gradio_call(f, extra_outputs=extra_outputs)

def initialize():
    modelloader.cleanup_models()
    modules.sd_models.setup_model()
    codeformer.setup_model(cmd_opts.codeformer_models_path)
    gfpgan.setup_model(cmd_opts.gfpgan_models_path)
    shared.face_restorers.append(modules.face_restoration.FaceRestoration())
    modelloader.load_upscalers()
    
    modules.scripts.load_scripts(os.path.join(script_path, "scripts"))
    
    shared.sd_model = modules.sd_models.load_model()
    shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: modules.sd_models.reload_model_weights(shared.sd_model)))
    shared.opts.onchange("sd_hypernetwork", wrap_queued_call(lambda: modules.hypernetworks.hypernetwork.load_hypernetwork(shared.opts.sd_hypernetwork)))
    shared.opts.onchange("sd_hypernetwork_strength", modules.hypernetworks.hypernetwork.apply_strength)

def find_sampler_ix_byname(sampler_name: str):
    found_sampler = 0
    sampler_name = sampler_name.lower()
    for ix, sampler in enumerate(sd_samplers.samplers):
        if sampler_name in sampler.aliases:
            found_sampler = ix
            break
    return found_sampler

# ^ Helpers / Init =================================================================================================================
# ^ ================================================================================================================================

# - ================================================================================================================================
# - Webserver ======================================================================================================================

app = Flask(__name__)
dev_mode = True

diffuse_settings_default = {
    'prompt': None,
    'negative_prompt': '',
    'prompt_style': '',
    'prompt_style2': '',
    'steps': 50,
    'sampler_index': 0,
    'restore_faces': False,
    'tiling': False,
    'n_iter': 1,
    'batch_size': 1,
    'cfg_scale': 7.5,
    'seed': -1,
    'subseed': None,
    'subseed_strength': None,
    'seed_resize_from_h': None,
    'seed_resize_from_w': None,
    'seed_enable_extras': None,
    'height': 512,
    'width': 512,
    'enable_hr': False,
    'denoising_strength': None,
    'firstphase_width': None,
    'firstphase_height': None
}

# GET /diffuse/Some%20Prompt%20Here?sampler=k_euler
@app.route("/diffuse/<prompt>", methods=["GET"])
def diffuse_get(prompt: str):
    print(f'Diffusing: [yellow]{prompt}[/]')
    
    if prompt is None or len(prompt.strip()) == 0:
        return Response('\'Prompt\' path argument is required.', status=400)
    
    diffuse_settings = diffuse_settings_default.copy()
    diffuse_settings['prompt'] = prompt
    
    override_params = request.args.to_dict()
    
    return run_diffusion(diffuse_settings, override_params)

# POST /diffuse
@app.route("/diffuse", methods=["POST"])
def diffuse_post():
    content_type = request.headers.get('Content-Type')
    
    if content_type != 'application/json':
        return Response('Content-Type not supported!', status=400)
    
    override_params = json.loads(request.data)
    
    if not 'prompt' in override_params:
        return Response('\'Prompt\' field is required.', status=400)
    
    print(f'Diffusing: [yellow]{override_params["prompt"]}[/]')
    return run_diffusion(diffuse_settings_default.copy(), override_params)

def run_diffusion(diffuse_settings: dict, override_params: dict = None):
    """
    Helper to cut down on boilerplate.
    diffuse_settings: dict of key/value pairs of diffusion settings.
    override_params: dict of key/value pairs that overwrite values in diffuse_settings if matching.
    """
    if override_params is None:
        override_params = {}
    
    # Override defaults in diffuse_settings if payload key name matches
    for k, v in override_params.items():
        if k in diffuse_settings:
            diffuse_settings[k] = v
    
    if 'sampler' in override_params:
        diffuse_settings['sampler_index'] = find_sampler_ix_byname(override_params['sampler'])
    
    result_images, result_info_js, result_info_html = modules.txt2img.txt2img(**diffuse_settings)
    
    # Hack: get first image response (could be many in the future), save to buffer in memory so server can send as BinaryIO
    first_image = result_images[0]
    img_io = BytesIO()
    first_image.save(img_io, 'png', compression=2)
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/png')

# - Webserver ======================================================================================================================
# - ================================================================================================================================

def run_webserver():
    print('[blue]Initializing Diffusion Model...[/]\n')
    initialize()
    sd_samplers.set_samplers()
    
    print('\n[blue]Starting Webserver...[/]')
    if dev_mode:
        print('[red]  In Developer Mode[/]\n')
        app.run(host="0.0.0.0", port=8080)
    else:
        print('[green]  In Production Mode[/]\n')
        from waitress import serve
        serve(app, host="0.0.0.0", port=8080)

if __name__ == "__main__":
    run_webserver()
