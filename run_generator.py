# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
import cv2

import pretrained_networks

#----------------------------------------------------------------------------

def generate_images(network_pkl, seeds, truncation_psi):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi

    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
        PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('seed%04d.png' % seed))

#----------------------------------------------------------------------------

def style_mixing_example(network_pkl, row_seeds, col_seeds, truncation_psi, col_styles, minibatch_size=4):
    print("Original")
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    w_avg = Gs.get_var('dlatent_avg') # [component]

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = minibatch_size

    print('Generating W vectors...')
    all_seeds = list(set(row_seeds + col_seeds))
    all_z = np.stack([np.random.RandomState(seed).randn(*Gs.input_shape[1:]) for seed in all_seeds]) # [minibatch, component]
    all_w = Gs.components.mapping.run(all_z, None) # [minibatch, layer, component]
    all_w = w_avg + (all_w - w_avg) * truncation_psi # [minibatch, layer, component]
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))} # [layer, component]

    print('Generating images...')
    all_images = Gs.components.synthesis.run(all_w, **Gs_syn_kwargs) # [minibatch, height, width, channel]
    image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}

    print('Generating style-mixed images...')
    for row_seed in row_seeds:
        for col_seed in col_seeds:
            w = w_dict[row_seed].copy()
            w[col_styles] = w_dict[col_seed][col_styles]
            image = Gs.components.synthesis.run(w[np.newaxis], **Gs_syn_kwargs)[0]
            image_dict[(row_seed, col_seed)] = image

    print('Saving images...')
    for (row_seed, col_seed), image in image_dict.items():
        PIL.Image.fromarray(image, 'RGB').save(dnnlib.make_run_dir_path('%d-%d.png' % (row_seed, col_seed)))

    print('Saving image grid...')
    _N, _C, H, W = Gs.output_shape
    canvas = PIL.Image.new('RGB', (W * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), 'white')
    for row_idx, row_seed in enumerate([None] + row_seeds):
        for col_idx, col_seed in enumerate([None] + col_seeds):
            if row_seed is None and col_seed is None:
                continue
            key = (row_seed, col_seed)
            if row_seed is None:
                key = (col_seed, col_seed)
            if col_seed is None:
                key = (row_seed, row_seed)
            canvas.paste(PIL.Image.fromarray(image_dict[key], 'RGB'), (W * col_idx, H * row_idx))
    canvas.save(dnnlib.make_run_dir_path('grid.png'))

#----------------------------------------------------------------------------

def style_mix_2(network_pkl, to_seeds, from_seeds, truncation_psi, col_styles, num_blends, minibatch_size=4):

    num_intermediates=num_blends
    print("Blending", num_intermediates)
    
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    w_avg = Gs.get_var('dlatent_avg') # [component]

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = minibatch_size

    print('Generating W vectors...')
    all_seeds = list(set(to_seeds + from_seeds))
    all_z = np.stack([np.random.RandomState(seed).randn(*Gs.input_shape[1:]) for seed in all_seeds]) # [minibatch, component]
    all_w = Gs.components.mapping.run(all_z, None) # [minibatch, layer, component]
    all_w = w_avg + (all_w - w_avg) * truncation_psi # [minibatch, layer, component]
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))} # [layer, component]

    print('Generating images...')
    all_images = Gs.components.synthesis.run(all_w, **Gs_syn_kwargs) # [minibatch, height, width, channel]
    image_dict = {(seed, seed, 0): image for seed, image in zip(all_seeds, list(all_images))}

    print('Generating style-mixed images...')
    for to_seed in to_seeds:

        to_w = w_dict[to_seed].copy()
        
        for from_seed in from_seeds:

            from_w = w_dict[from_seed].copy()
            
            for i in range(num_intermediates):

                new_w = from_w
                new_w[col_styles] = ((i+1)/(num_intermediates+1) * to_w[col_styles]) + ((1-(i+1)/(num_intermediates+1)) * from_w[col_styles])
                
                image = Gs.components.synthesis.run(new_w[np.newaxis], **Gs_syn_kwargs)[0]
                image_dict[(to_seed, from_seed, i)] = image

    print('Saving images...')
    for (row_seed, col_seed, i), image in image_dict.items():
        PIL.Image.fromarray(image, 'RGB').save(dnnlib.make_run_dir_path('%d-%d-%d.png' % (row_seed, col_seed, i)))

    print('Saving image grid...')
    _N, _C, H, W = Gs.output_shape
    canvas = PIL.Image.new('RGB', (W * (num_intermediates + 2), H * (len(to_seeds)*len(from_seeds))), 'black')
    
    for from_idx, from_seed in enumerate(from_seeds):
        for to_idx, to_seed in enumerate(to_seeds):

            fro = (from_seed, from_seed,0)
            to = (to_seed, to_seed, 0)
            canvas.paste(PIL.Image.fromarray(image_dict[fro], 'RGB'), (0, H * (to_idx+from_idx*len(to_seeds))))
            canvas.paste(PIL.Image.fromarray(image_dict[to], 'RGB'), (W * (num_intermediates+1), H * (to_idx+from_idx*len(to_seeds))))
            
            for i in range(num_intermediates):
                key = (to_seed, from_seed,i)
                canvas.paste(PIL.Image.fromarray(image_dict[key], 'RGB'), (W * (i+1), H * (to_idx+from_idx*len(to_seeds))))
                
    canvas.save(dnnlib.make_run_dir_path('grid.png'))

#----------------------------------------------------------------------------

def style_mix_3(network_pkl, from_seeds, to_seeds, truncation_psi, col_styles, num_blends, view_size, minibatch_size=4):

    global mix_val
    global from_val
    global to_val
    global psi_val
    global styles
    
    mix_val = 0
    psi_val = truncation_psi
    styles = col_styles
    
    num_intermediates=num_blends
    windowName = "Mix Result"

    
    def from_change(value):    

        global from_val
        from_val = from_seeds[value]
        on_change()
        
    def to_change(value):    

        global to_val
        to_val = to_seeds[value]
        on_change()

    def mix_change(value):    

        global mix_val
        mix_val = value
        on_change()

    def psi_change(value):    

        global psi_val
        psi_val = value
        on_change()

        
    def on_change():    

        global mix_val
        global from_val
        global to_val
        global psi_val
        global styles
        
        from_w = w_dict[from_val].copy()   
        to_w = w_dict[to_val].copy()

        new_w = from_w
        new_w[styles] = (mix_val/num_intermediates * to_w[styles]) + ((num_intermediates-mix_val)/num_intermediates * from_w[styles])

        image = Gs.components.synthesis.run(new_w[np.newaxis], **Gs_syn_kwargs)[0]
        imageCopy = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
        imageCopy = cv2.resize(imageCopy, (view_size,view_size))
        
        cv2.imshow(windowName, imageCopy)
        
        return
    
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    w_avg = Gs.get_var('dlatent_avg') # [component]

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = minibatch_size

    print('Generating W vectors...')
    all_seeds = list(set(to_seeds + from_seeds))
    all_z = np.stack([np.random.RandomState(seed).randn(*Gs.input_shape[1:]) for seed in all_seeds]) # [minibatch, component]
    all_w = Gs.components.mapping.run(all_z, None) # [minibatch, layer, component]
    all_w = w_avg + (all_w - w_avg) * truncation_psi # [minibatch, layer, component]
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))} # [layer, component]

    print('Generating images...')
    all_images = Gs.components.synthesis.run(all_w, **Gs_syn_kwargs) # [minibatch, height, width, channel]
    image_dict = {(seed, seed, 0): image for seed, image in zip(all_seeds, list(all_images))}

    from_w = w_dict[from_seeds[0]].copy()
    to_w = w_dict[to_seeds[0]].copy()
    from_val = from_seeds[0]
    to_val = to_seeds[0]

    image = Gs.components.synthesis.run(from_w[np.newaxis], **Gs_syn_kwargs)[0]
                                     
    image_dict[(to_seeds[0], to_seeds[0], 0)] = image
    image_cv2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_cv2 = cv2.resize(image_cv2, (view_size,view_size))
    cv2.imshow(windowName, image_cv2)
    
    if len(from_seeds) > 1:
        cv2.createTrackbar('from', windowName, 0, len(from_seeds)-1, from_change)    
    if len(to_seeds) > 1:
        cv2.createTrackbar('to', windowName, 0, len(to_seeds)-1, to_change)
    cv2.createTrackbar('mix', windowName, 0, num_blends, mix_change)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('Saving images...')
    print("From seed", from_val, "To seed", to_val, "Mix", mix_val)
    from_w = w_dict[from_val].copy()
    to_w = w_dict[to_val].copy()  
    new_w = from_w
    new_w[styles] = (mix_val/num_intermediates * to_w[styles]) + ((num_intermediates-mix_val)/num_intermediates * from_w[styles])
    image = Gs.components.synthesis.run(new_w[np.newaxis], **Gs_syn_kwargs)[0]
    PIL.Image.fromarray(image, 'RGB').save(dnnlib.make_run_dir_path('%d-%d-%d.png' % (from_val, to_val, mix_val)))



    
#----------------------------------------------------------------------------

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

_examples = '''examples:

  # Generate ffhq uncurated images (matches paper Figure 12)
  python %(prog)s generate-images --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --seeds=6600-6625 --truncation-psi=0.5

  # Generate ffhq curated images (matches paper Figure 11)
  python %(prog)s generate-images --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --seeds=66,230,389,1518 --truncation-psi=1.0

  # Generate uncurated car images (matches paper Figure 12)
  python %(prog)s generate-images --network=gdrive:networks/stylegan2-car-config-f.pkl --seeds=6000-6025 --truncation-psi=0.5

  # Generate style mixing example (matches style mixing video clip)
  python %(prog)s style-mixing-example --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --row-seeds=85,100,75,458,1500 --col-seeds=55,821,1789,293 --truncation-psi=1.0
'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='''StyleGAN2 generator.

Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    parser_generate_images = subparsers.add_parser('generate-images', help='Generate images')
    parser_generate_images.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_generate_images.add_argument('--seeds', type=_parse_num_range, help='List of random seeds', required=True)
    parser_generate_images.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_generate_images.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    parser_style_mixing_example = subparsers.add_parser('style-mixing-example', help='Generate style mixing video')
    parser_style_mixing_example.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_style_mixing_example.add_argument('--row-seeds', type=_parse_num_range, help='Random seeds to use for image rows', required=True)
    parser_style_mixing_example.add_argument('--col-seeds', type=_parse_num_range, help='Random seeds to use for image columns', required=True)
    parser_style_mixing_example.add_argument('--col-styles', type=_parse_num_range, help='Style layer range (default: %(default)s)', default='0-6')
    parser_style_mixing_example.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_style_mixing_example.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    parser_style_mix_2 = subparsers.add_parser('style-mix-2', help='Generate style mix 2')
    parser_style_mix_2.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_style_mix_2.add_argument('--to-seeds', type=_parse_num_range, help='Random seeds to use for to images', required=True)
    parser_style_mix_2.add_argument('--from-seeds', type=_parse_num_range, help='Random seeds to use for from images', required=True)
    parser_style_mix_2.add_argument('--col-styles', type=_parse_num_range, help='Style layer range (default: %(default)s)', default='0-13')
    parser_style_mix_2.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_style_mix_2.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    parser_style_mix_2.add_argument('--num-blends', type=int, help='Number of intermediate blends (default: %(default)s)', default=10)

    parser_style_mix_3 = subparsers.add_parser('style-mix-3', help='Generate style mix 3')
    parser_style_mix_3.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_style_mix_3.add_argument('--from-seeds', type=_parse_num_range, help='Random seeds to use for mapping', required=True)
    parser_style_mix_3.add_argument('--to-seeds', type=_parse_num_range, help='Random seed to use to map to', required=True)
    parser_style_mix_3.add_argument('--col-styles', type=_parse_num_range, help='Style layer range (default: %(default)s)', default='0-13')
    parser_style_mix_3.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_style_mix_3.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    parser_style_mix_3.add_argument('--num-blends', type=int, help='Number of intermediate blends (default: %(default)s)', default=100)
    parser_style_mix_3.add_argument('--view_size', type=int, help='Display size for image (default: %(default)s)', default=512)

    args = parser.parse_args()
    kwargs = vars(args)
    subcmd = kwargs.pop('command')

    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = subcmd

    func_name_map = {
        'generate-images': 'run_generator.generate_images',
        'style-mixing-example': 'run_generator.style_mixing_example',
        'style-mix-2': 'run_generator.style_mix_2',
        'style-mix-3': 'run_generator.style_mix_3'
        }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
